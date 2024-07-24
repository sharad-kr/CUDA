#include<stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>
#include <dirent.h>
float* kernel_conv1 = (float*) malloc(sizeof(float) * 20 * (5*5));
float* bias_conv1 = (float*) malloc(sizeof(float) * 20);
float* kernel_conv2 = (float*) malloc(sizeof(float) * 20 * 50 * (5*5));
float* bias_conv2 = (float*) malloc(sizeof(float) * 50);
float* kernel_fc1 = (float*) malloc(sizeof(float) * 50 * 500 * (4*4));
float* bias_fc1 = (float*) malloc(sizeof(float) * 500);
float* kernel_fc2 = (float*) malloc(sizeof(float) * 500 * 10 * (1*1));;
float* bias_fc2 = (float*) malloc(sizeof(float) * 10 );
float* images = (float*) malloc(sizeof(float) * 10000 * 28*28);
char* filenames[10000];

float* d_kernel_conv1;
float* d_bias_conv1;
float* d_kernel_conv2;
float* d_bias_conv2;
float* d_kernel_fc1;
float* d_bias_fc1;
float* d_kernel_fc2;
float* d_bias_fc2;


__global__ void global_softmax(int vector_length, float* d_input){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i==0){
        float sum = 0;
        for(int j = 0 ; j<10 ; j++){
            sum += exp(d_input[j]);
            
        }
        for(int j = 0 ; j<10 ; j++){
            d_input[j] = exp(d_input[j])/sum;
        }
        
    }

}

__global__ void global_storeAns(int image_index, float* d_ans_probabilities, float* d_layer6_nodes){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<10){
        d_ans_probabilities[image_index*10 + i] = d_layer6_nodes[i];
    }
}

__global__ void global_cudaMaxPool(int num_inputChannel, int order_inputMatrix, float* d_inputLayer, float* d_outputLayer){


    int order_outputMatrix = order_inputMatrix/2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.z * blockDim.z + threadIdx.z;   


            if(i<num_inputChannel && x<order_inputMatrix && y<order_inputMatrix && x%2==0 && y%2==0){

                    float maxi =d_inputLayer[i*order_inputMatrix*order_inputMatrix + x*order_inputMatrix + y];
                    for(int p = x ; p<x+2 ; p++){
                        for(int k = y ; k<y+2 ; k++){
                            if(maxi < d_inputLayer[i*order_inputMatrix*order_inputMatrix + p*order_inputMatrix + k]) maxi = d_inputLayer[i*order_inputMatrix*order_inputMatrix + p*order_inputMatrix + k];
                        }
                    }
                    d_outputLayer[i*order_outputMatrix*order_outputMatrix + (x/2)*order_outputMatrix + (y/2)] = maxi;
                }
    }


__global__ void global_cudaConvolution4D(int num_inputChannel, int num_outputChannel, int order_inputMatrix, int order_kernel, float* d_inputLayer, float* d_weights,float* d_outputLayer, float* d_bias){

    int order_outputMatrix = order_inputMatrix - order_kernel + 1;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    int x = w/order_outputMatrix;
    int y = w % order_outputMatrix;

    if(i<num_outputChannel && j<num_inputChannel && x<order_outputMatrix && y<order_outputMatrix){                

                            float sum = 0;
                            for(int p = 0 ; p<order_kernel ; p++){
                                for(int k = 0 ; k<order_kernel ; k++){
                                    sum += d_weights[ i*num_inputChannel*order_kernel*order_kernel + j*(order_kernel*order_kernel) + p*order_kernel + k] * d_inputLayer[j*order_inputMatrix*order_inputMatrix + (x+p)*order_inputMatrix + (y+k)];
                                }
                            }
                            atomicAdd(&d_outputLayer[i*order_outputMatrix*order_outputMatrix + x*order_outputMatrix + y],sum);

            }

    if(j==0){
        atomicAdd(&d_outputLayer[i*order_outputMatrix*order_outputMatrix + x*order_outputMatrix + y],d_bias[i]);
    }

}


__global__ void global_cudaImageToFirstConv(int num_inputChannel, int num_outputChannel, int order_inputMatrix, int order_kernel,int image_index ,float* d_inputLayer, float* d_weights,float* d_outputLayer, float* d_bias){

    int order_outputMatrix = order_inputMatrix - order_kernel + 1;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    int x = w/order_outputMatrix;
    int y = w % order_outputMatrix;

    if(i<num_outputChannel && j<num_inputChannel && x<order_outputMatrix && y<order_outputMatrix){                

                            float sum = 0;
                            for(int p = 0 ; p<order_kernel ; p++){
                                for(int k = 0 ; k<order_kernel ; k++){
                                    sum += d_weights[ i*num_inputChannel*order_kernel*order_kernel + j*(order_kernel*order_kernel) + p*order_kernel + k] * d_inputLayer[image_index*28*28 + j*order_inputMatrix*order_inputMatrix + (x+p)*order_inputMatrix + (y+k)];
                                }
                            }
                            atomicAdd(&d_outputLayer[i*order_outputMatrix*order_outputMatrix + x*order_outputMatrix + y],sum);

            }

    if(j==0){
        atomicAdd(&d_outputLayer[i*order_outputMatrix*order_outputMatrix + x*order_outputMatrix + y],d_bias[i]);
    }

}


__global__ void global_cudaReLU(int num_inputChannel, float* d_inputLayer){



    int i = blockIdx.x * blockDim.x + threadIdx.x;;
    int x = 0;
    int y = 0;

    if(i<num_inputChannel){
        if(d_inputLayer[i*1*1 + x*1 + y] < 0) d_inputLayer[i*1*1 + x*1 + y] = 0;
    }
}


// void printTop5Probabilities(float* probabilities, int size) {
//     // Create an array to store the indices of probabilities
//     int indices[size];
//     for (int i = 0; i < size; i++) {
//         indices[i] = i;
//     }

//     // Sort indices based on probabilities
//     for (int i = 0; i < size - 1; i++) {
//         for (int j = 0; j < size - i - 1; j++) {
//             if (probabilities[j] < probabilities[j + 1]) {
//                 float temp = probabilities[j];
//                 probabilities[j] = probabilities[j + 1];
//                 probabilities[j + 1] = temp;

//                 int tempIndex = indices[j];
//                 indices[j] = indices[j + 1];
//                 indices[j + 1] = tempIndex;
//             }
//         }
//     }

//     // Print top 5 probabilities with their classes
//     // printf("Top 5 Probabilities:\n");
//     for (int i = 0; i < 5; i++) {
//         printf("%.6f class %d\n", probabilities[i], indices[i]);
//     }
// }


void readFiles2(const char* filename, int index){

        FILE* file;

        file = fopen(filename,"r");

        for(int i = 0 ; i< 28 ; i++){
            for(int j = 0 ; j< 28 ; j++){
                fscanf(file,"%f ", &images[28*28*index+i*28 + j]);
            }
        }

        fclose(file);


}


void readFiles(){

        FILE* file;




        file = fopen("../weights/conv1.txt","r");

        for(int i = 0 ; i<20 ; i++){

            for(int p = 0 ; p<5 ; p++){
                for(int k = 0 ; k<5 ; k++){
                    fscanf(file,"%f ",&kernel_conv1[i*25 + p*5 + k]);
                }
            }
        }



        for(int i = 0 ; i<20 ; i++){
            fscanf(file,"%f ",&bias_conv1[i]);
        }

        file = fopen("../weights/conv2.txt","r");

        for(int i = 0 ; i<1000 ; i++){

            for(int p = 0 ; p<5 ; p++){
                for(int k = 0 ; k<5 ; k++){
                    fscanf(file,"%f ",&kernel_conv2[i*25 + p*5 + k]);
                }
            }
        }

        for(int i = 0 ; i<50 ; i++){
            fscanf(file,"%f ",&bias_conv2[i]);
        }


        file = fopen("../weights/fc1.txt","r");

        for(int i = 0 ; i<25000 ; i++){

            for(int p = 0 ; p<4 ; p++){
                for(int k = 0 ; k<4 ; k++){
                    fscanf(file,"%f ",&kernel_fc1[i*16 + p*4 + k]);
                }
            }
        }

        for(int i = 0 ; i<500 ; i++){
            fscanf(file,"%f ",&bias_fc1[i]);
        }



        file = fopen("../weights/fc2.txt","r");

        for(int i = 0 ; i<5000 ; i++){
            // float** matrix = kernel_fc2[i];
            for(int p = 0 ; p<1 ; p++){
                for(int k = 0 ; k<1 ; k++){
                    fscanf(file,"%f ",&kernel_fc2[i*1*1 + p*1 +k]);
                }
            }
        }



        for(int i = 0 ; i<10 ; i++){
            fscanf(file,"%f ",&bias_fc2[i]);
        }

        fclose(file);
}

void cudaCopyParameters(){

        cudaMalloc((void**)&d_kernel_conv1, sizeof(float) * 20 * (5*5));
        cudaMalloc((void**)&d_bias_conv1, sizeof(float) * 20);

        cudaMalloc((void**)&d_kernel_conv2, sizeof(float) * 20 * 50 * (5*5));
        cudaMalloc((void**)&d_bias_conv2, sizeof(float) * 50);

        cudaMalloc((void**)&d_kernel_fc1, sizeof(float) * 50 * 500 * (4*4));
        cudaMalloc((void**)&d_bias_fc1, sizeof(float) * 500);

        cudaMalloc((void**)&d_kernel_fc2, sizeof(float) * 500 * 10 * (1*1));
        cudaMalloc((void**)&d_bias_fc2, sizeof(float) * 10);



        cudaMemcpy(d_kernel_conv1, kernel_conv1, sizeof(float) * 20 * (5*5),cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias_conv1, bias_conv1, sizeof(float) * 20,cudaMemcpyHostToDevice );

        cudaMemcpy(d_kernel_conv2, kernel_conv2, sizeof(float) * 20 * 50 * (5*5),cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias_conv2, bias_conv2, sizeof(float) * 50 ,cudaMemcpyHostToDevice);

        cudaMemcpy(d_kernel_fc1, kernel_fc1, sizeof(float) * 50 * 500 * (4*4),cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias_fc1, bias_fc1, sizeof(float) * 500,cudaMemcpyHostToDevice );

        cudaMemcpy(d_kernel_fc2, kernel_fc2, sizeof(float) * 500 * 10 * (1*1),cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias_fc2, bias_fc2, sizeof(float) * 10,cudaMemcpyHostToDevice );

}

void freeMemory(){

        free(images);
        free(kernel_conv1);
        free(kernel_conv2);
        free(kernel_fc1);
        free(kernel_fc2);
        free(bias_conv1);
        free(bias_conv2);
        free(bias_fc1);
        free(bias_fc2);

}



void inference(int num_images, float* images){

    float* d_images;
    float* d_ans_probabilities;
    cudaMalloc((void**)&d_ans_probabilities, sizeof(float) * 10 * num_images);
    cudaMemset(d_ans_probabilities, 0, 10 * num_images * sizeof(float));
    cudaMalloc((void**)&d_images, sizeof(float)* num_images * 28 * 28);
    cudaMemcpy(d_images, images, sizeof(float) * num_images * 28 * 28,cudaMemcpyHostToDevice);

    // float* d_input_image;
    float* d_layer1_nodes;
    float* d_layer2_nodes;
    float* d_layer3_nodes;
    float* d_layer4_nodes;
    float* d_layer5_nodes;
    float* d_layer6_nodes;
    float* h_ans_probabilities = (float*) calloc(10 * (num_images),sizeof(float));
    

    
    // cudaMalloc((void**)&d_input_image, sizeof(float) * 28 * 28);
    cudaMalloc((void**)&d_layer1_nodes, sizeof(float) * 20 * (24*24));
    cudaMalloc((void**)&d_layer2_nodes, sizeof(float) * 20 * (12*12));
    cudaMalloc((void**)&d_layer3_nodes, sizeof(float) * 50 * (8*8));
    cudaMalloc((void**)&d_layer4_nodes, sizeof(float) * 50 * (4*4));
    cudaMalloc((void**)&d_layer5_nodes, sizeof(float) * 500 * (1*1));
    cudaMalloc((void**)&d_layer6_nodes, sizeof(float) * 10 * (1*1));
    
    
    // float* curr_image = (float*) malloc(28 * 28 * sizeof(float));
    // for(int i = 0 ; i<28 ; i++){
    //     for(int j = 0 ; j<28 ; j++){
    //         curr_image[i*28+j] = images[image_index*28*28 + i*28 + j];
    //     }
    // }
    // cudaMemcpy(d_input_image, curr_image, sizeof(float) * 28 * 28 ,cudaMemcpyHostToDevice);
for(int image_index = 0 ; image_index<num_images ; image_index++)
        
{
    cudaMemset(d_layer1_nodes, 0, 20 * (24*24) * sizeof(float));
    cudaMemset(d_layer2_nodes, 0, 20 * (12*12) * sizeof(float));
    cudaMemset(d_layer3_nodes, 0, 50 * (8*8) * sizeof(float));
    cudaMemset(d_layer4_nodes, 0, 50 * (4*4) * sizeof(float));
    cudaMemset(d_layer5_nodes, 0, 500 * (1*1) * sizeof(float));
    cudaMemset(d_layer6_nodes, 0, 10 * (1*1) * sizeof(float));

    dim3 blockDim1(20,1,48);
    dim3 gridDim1(1,1,12);
    global_cudaImageToFirstConv<<<gridDim1,blockDim1>>>(1, 20, 28, 5,image_index ,d_images, d_kernel_conv1, d_layer1_nodes,d_bias_conv1);


    dim3 blockDim2(20,12,4);
    dim3 gridDim2(1,2,6);
    global_cudaMaxPool<<<gridDim2,blockDim2>>>(20, 24,  d_layer1_nodes,  d_layer2_nodes);


    dim3 blockDim3(10,4,8);
    dim3 gridDim3(5,5,8);
    global_cudaConvolution4D<<<gridDim3,blockDim3>>>(20, 50, 12, 5, d_layer2_nodes, d_kernel_conv2, d_layer3_nodes,d_bias_conv2);
   
    dim3 blockDim4(50,4,4);
    dim3 gridDim4(1,2,2);
    global_cudaMaxPool<<<gridDim4,blockDim4>>>(50, 8,  d_layer3_nodes,  d_layer4_nodes);

    dim3 blockDim5(25,25,1);
    dim3 gridDim5(21,2,1);
    global_cudaConvolution4D<<<gridDim5,blockDim5>>>(50, 500, 4, 4, d_layer4_nodes, d_kernel_fc1, d_layer5_nodes,d_bias_fc1);

    dim3 blockDim6(500,1,1);
    dim3 gridDim6(1,1,1);
    global_cudaReLU<<<gridDim6,blockDim6>>>(500, d_layer5_nodes);
   
    dim3 blockDim7(6,101,1);
    dim3 gridDim7(2,5,1);
    global_cudaConvolution4D<<<gridDim7,blockDim7>>>(500, 10, 1, 1, d_layer5_nodes, d_kernel_fc2, d_layer6_nodes,d_bias_fc2); 

    dim3 blockDim8(2,1,1);
    dim3 gridDim8(1,1,1);
    global_softmax<<<gridDim8,blockDim8>>>(10,d_layer6_nodes); 
    
    

    dim3 blockDim9(10,1,1);
    dim3 gridDim9(1,1,1);
    global_storeAns<<<gridDim9,blockDim9>>>(image_index, d_ans_probabilities, d_layer6_nodes);
}
    
    cudaMemcpy(h_ans_probabilities,d_ans_probabilities ,10 * (num_images) * sizeof(float), cudaMemcpyDeviceToHost); 







    FILE* outputFile;
    char o_filePath[200];

    for (int image_index = 0; image_index < num_images; image_index++) {
        float probabilities[10];

        // Initialize probabilities array with values from h_ans_probabilities
        for (int i = 0; i < 10; i++) {
            probabilities[i] = h_ans_probabilities[image_index * 10 + i];
        }

        int indices[10];
        for (int i = 0; i < 10; i++) {
            indices[i] = i;
        }

        // Sort indices based on probabilities
        for (int i = 0; i < 10 - 1; i++) {
            for (int j = 0; j < 10 - i - 1; j++) {
                if (probabilities[j] < probabilities[j + 1]) {
                    float temp_prob = probabilities[j];
                    probabilities[j] = probabilities[j + 1];
                    probabilities[j + 1] = temp_prob;

                    int temp_index = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = temp_index;
                }
            }
        }

        // Print top 5 probabilities with their classes to file
        sprintf(o_filePath, "../output/%s", filenames[image_index]);
        outputFile = fopen(o_filePath, "w");
        if (outputFile != NULL) {
            for (int i = 0; i < 5; i++) {
                fprintf(outputFile, "%.6f Class %d\n", probabilities[i], indices[i]);
            }
            fclose(outputFile);
        } else {
            printf("Error opening file for writing: %s\n", o_filePath);
        }
    }

//    float dr = 0;

//     for(int i = 0 ; i<10 ; i++){
//        dr+=h_layer6_nodes[i];
//     }

//     for(int i = 0 ; i<10 ; i++){
//         h_layer6_nodes[i] = (float)h_layer6_nodes[i]/dr;
//         }
    

//    int taken[10] = {0};
//     float top5Prob[5] = {0};
//     int top5Class[5] = {0};

//     for (int i = 0; i < 5; i++) {
//         float max_value = -1;
//         int max_index = -1;
//         for (int j = 0; j < 10; j++) {
//             if (h_layer6_nodes[j] > max_value) {
//                 max_value = h_layer6_nodes[j];
//                 max_index = j;
//             }
//         }
//         top5Prob[i] = max_value;
//         top5Class[i] = max_index;
//         h_layer6_nodes[max_index] = -1; 
//     }

//     for (int i = 0; i < 5; i++) {
//         printf("%f Class %d\n", top5Prob[i], top5Class[i]);
//     }

    // cudaFree(d_input_image);
    cudaFree(d_layer1_nodes);
    cudaFree(d_layer2_nodes);
    cudaFree(d_layer3_nodes);
    cudaFree(d_layer4_nodes);
    cudaFree(d_layer5_nodes);
    cudaFree(d_layer6_nodes);

}



void read_folder(const char* folder){
    DIR *dir;
    struct dirent *entry;
    // Open directory
    dir = opendir(folder);
    if (dir == NULL) {
        perror("Unable to open directory");
        exit(EXIT_FAILURE);
    }
    int count=0;
    while ((entry = readdir(dir)) != NULL){
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        char filepath[100];
        char filename[100];
        strcpy(filepath, folder);
        strcat(filepath, "/");
        strcat(filepath, entry->d_name);
        strcpy(filename, entry->d_name);
        filenames[count] = (char*) malloc((1+strlen(filename)) * sizeof(char));
        memcpy(filenames[count], &filename, strlen(filename) + 1);
        
        // filenames[count] = filename;
        printf("Reading %s\n", filename);
        readFiles2(filepath,count);
        count+=1;
        if (count==10000){
            break;
        }
    }
}




int main() {

struct timespec startTime,timeToCopyParam ,endTime;
readFiles();
read_folder("../pre-proc-img");


clock_gettime(CLOCK_MONOTONIC, &startTime);

cudaCopyParameters();

clock_gettime(CLOCK_MONOTONIC, &timeToCopyParam);

inference(10000,images);

// for(int i = 0 ; i<70 ; i++){
//     printf ("Image %d\n", i);
//     inference(i,images);
// }


clock_gettime(CLOCK_MONOTONIC, &endTime);

long paramCopy_timeDifference ,timeDifference;

paramCopy_timeDifference = (timeToCopyParam.tv_sec - startTime.tv_sec ) * 1000000000L +
                 (timeToCopyParam.tv_nsec - startTime.tv_nsec);

timeDifference = (endTime.tv_sec - timeToCopyParam.tv_sec) * 1000000000L +
                 (endTime.tv_nsec - timeToCopyParam.tv_nsec);

double paramCopy_milliseconds = (double)paramCopy_timeDifference / 1000000.0;
double process_milliseconds = (double)timeDifference / 1000000.0;


printf("\n\nTime taken to copy parameters(CPU -> GPU): %.2f milliseconds\n", paramCopy_milliseconds);
printf("Time taken for inference: %.2f milliseconds\n", process_milliseconds);



freeMemory();

}

