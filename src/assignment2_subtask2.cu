#include<stdio.h>
#include<math.h>



__global__ void convolution(int mRow, int mCol, int knRow, int knCol,float* d_kernel, float* d_matrix, float* d_ans, float bias ){
    int ansRow = mRow - knRow + 1;
    int ansCol = mCol - knCol + 1;

    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<ansRow && j<ansCol){
            float sum = 0;
            for(int p = 0 ; p<knRow ; p++){
                for(int k = 0 ; k<knCol ; k++){
                    sum += d_kernel[p*knCol + k] * d_matrix[(i+p)*mCol + (j+k)];
                }
            }
            d_ans[i*ansCol + j] = sum + bias;
        }
}

__global__ void maxPooling(int mRow, int mCol,int pooling_size ,float* d_matrix, float* d_ans){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if(i < mRow && j < mCol && i%pooling_size==0 && j%pooling_size==0){

 

                float maxi = d_matrix[i*mCol + j];
                for(int p = i ; p<i+pooling_size ; p++){
                    for(int k = j ; k<j+pooling_size ; k++){
                        if(maxi < d_matrix[p*mCol + k]) maxi = d_matrix[p*mCol + k];
                    }
                }
                d_ans[(i/pooling_size)*(mCol/pooling_size) + j/pooling_size] = maxi;




        }
}

__global__ void avgPooling(int mRow, int mCol,int pooling_size ,float* d_matrix, float* d_ans){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if(i < mRow && j < mCol && i%pooling_size==0 && j%pooling_size==0){

 

            float sum = 0;
            for(int p = i ; p<i+pooling_size ; p++){
                for(int k = j ; k<j+pooling_size ; k++){
                    sum += d_matrix[p*mCol + k];
                }
            }
            d_ans[(i/pooling_size)*(mCol/pooling_size) + j/pooling_size] = sum/(pooling_size*pooling_size);





        }
}

__global__ void addMatrix(int row, int col, float* d_a, float* d_b){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
            if(i<row && j<col){  
                d_a[i*col + j] = d_a[i*col + j] + d_b[i*col + j];
            }
}

__global__ void ReLU(int row, int col, float* d_matrix){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<row && j<col){
        if(d_matrix[i*col + j] < 0) d_matrix[i*col + j] = 0;
    }
}

__global__ void tanH(int row, int col, float* d_matrix){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<row && j<col){

                float temp = d_matrix[i*col + j];
                temp = (exp(temp)-exp(-temp))/(exp(temp)+exp(-temp));
                d_matrix[i*col + j] = temp;

    }
}

void cudaConvolution(int mRow, int mCol, int knRow, int knCol, float** kernel, float** matrix, float** ans, float bias) {
    int ansRow = mRow - knRow + 1;
    int ansCol = mCol - knCol + 1;

    float* d_kernel;
    float* d_matrix;
    float* d_ans;

    cudaMalloc((void**)&d_kernel, knRow * knCol * sizeof(float));
    cudaMalloc((void**)&d_matrix, mRow * mCol * sizeof(float));
    cudaMalloc((void**)&d_ans, ansRow * ansCol * sizeof(float));


    float* h_kernel = (float*) malloc(knRow * knCol * sizeof(float));
    float* h_matrix = (float*) malloc(mRow * mCol * sizeof(float));
    float* h_ans = (float*) malloc(ansRow * ansCol * sizeof(float));

    for(int i = 0 ; i<knRow ;i++){
        for(int j = 0 ; j<knCol ; j++){
            h_kernel[i*knCol + j] = kernel[i][j];
        }
    }

    for(int i = 0 ; i<mRow ;i++){
        for(int j = 0 ; j<mCol ; j++){
            h_matrix[i*mCol + j] = matrix[i][j];
        }
    }


   
        cudaMemcpy(d_matrix, h_matrix, mRow * mCol * sizeof(float),cudaMemcpyHostToDevice );
    



    
        cudaMemcpy(d_kernel , h_kernel, knRow * knCol * sizeof(float),cudaMemcpyHostToDevice );




    dim3 blockSize(16, 16);
    dim3 gridSize((ansCol + blockSize.x - 1) / blockSize.x, (ansRow + blockSize.y - 1) / blockSize.y);

    convolution<<<gridSize, blockSize>>>(mRow, mCol, knRow, knCol, d_kernel, d_matrix, d_ans, bias);


        
    cudaMemcpy(h_ans, d_ans  , ansRow * ansCol * sizeof(float), cudaMemcpyDeviceToHost);          
        
   


  for(int i = 0 ; i<ansRow ; i++){
    for(int j = 0 ; j<ansCol ; j++){
        ans[i][j] = h_ans[i*ansCol + j];
    }
  }

    free(h_ans);
    free(h_kernel);
    free(h_matrix);

    cudaFree(d_kernel);
    cudaFree(d_matrix);
    cudaFree(d_ans);
}

void cudaMaxPooling(int mRow, int mCol, int pooling_size ,float** matrix, float** ans){
 

    float* d_matrix;
    float* d_ans;

    cudaMalloc((void**)&d_matrix, mRow * mCol * sizeof(float));
    cudaMalloc((void**)&d_ans, (mRow/pooling_size) * (mCol/pooling_size) * sizeof(float));

    float* h_matrix = (float*) malloc(mRow * mCol * sizeof(float));
    float* h_ans = (float*) malloc((mRow/pooling_size) * (mCol/pooling_size) * sizeof(float));    

    for(int i = 0 ; i<mRow ;i++){
        for(int j = 0 ; j<mCol ; j++){
            h_matrix[i*mCol + j] = matrix[i][j];
        }
    }


    cudaMemcpy(d_matrix, h_matrix, mRow * mCol * sizeof(float),cudaMemcpyHostToDevice );
    dim3 blockSize(16, 16);
    dim3 gridSize(((mCol) + blockSize.x - 1) / blockSize.x, ((mRow) + blockSize.y - 1) / blockSize.y);
    maxPooling<<<gridSize,blockSize>>>(mRow, mCol,pooling_size ,d_matrix, d_ans);
    cudaMemcpy(h_ans, d_ans  , (mRow/pooling_size) * (mCol/pooling_size) * sizeof(float), cudaMemcpyDeviceToHost);
       for(int i = 0 ; i<mRow/pooling_size ; i++){
        for(int j = 0 ; j<mCol/pooling_size ; j++){
            ans[i][j] = h_ans[i*(mCol/pooling_size) + j];
        }
      }   

    free(h_ans);
    free(h_matrix);

    cudaFree(d_matrix);
    cudaFree(d_ans);

}


void cudaAvgPooling(int mRow, int mCol, int pooling_size ,float** matrix, float** ans){
 

    float* d_matrix;
    float* d_ans;

    cudaMalloc((void**)&d_matrix, mRow * mCol * sizeof(float));
    cudaMalloc((void**)&d_ans, (mRow/pooling_size) * (mCol/pooling_size) * sizeof(float));

    float* h_matrix = (float*) malloc(mRow * mCol * sizeof(float));
    float* h_ans = (float*) malloc((mRow/pooling_size) * (mCol/pooling_size) * sizeof(float));    

    for(int i = 0 ; i<mRow ;i++){
        for(int j = 0 ; j<mCol ; j++){
            h_matrix[i*mCol + j] = matrix[i][j];
        }
    }


    cudaMemcpy(d_matrix, h_matrix, mRow * mCol * sizeof(float),cudaMemcpyHostToDevice );
    dim3 blockSize(16, 16);
    dim3 gridSize(((mCol) + blockSize.x - 1) / blockSize.x, ((mRow) + blockSize.y - 1) / blockSize.y);
    avgPooling<<<gridSize,blockSize>>>(mRow, mCol, pooling_size,d_matrix, d_ans);
    cudaMemcpy(h_ans, d_ans  , (mRow/pooling_size) * (mCol/pooling_size) * sizeof(float), cudaMemcpyDeviceToHost);
       for(int i = 0 ; i<mRow/pooling_size ; i++){
        for(int j = 0 ; j<mCol/pooling_size ; j++){
            ans[i][j] = h_ans[i*(mCol/pooling_size) + j];
        }
      }   

    free(h_ans);
    free(h_matrix);

    cudaFree(d_matrix);
    cudaFree(d_ans);

}

void cudaAddMatrix(int row, int col ,float** a , float** b ){
    
    float* d_a;
    float* d_b;

    cudaMalloc((void**)&d_a, row * col * sizeof(float));
    cudaMalloc((void**)&d_b, row * col * sizeof(float));
    float* h_a = (float*) malloc(row * col* sizeof(float));
    float* h_b = (float*) malloc(row * col * sizeof(float)); 
    float* h_ans = (float*) malloc(row * col * sizeof(float));  

    for(int i = 0 ; i<row ;i++){
        for(int j = 0 ; j<col ; j++){
            h_a[i*col + j] = a[i][j];
        }
    }  
    for(int i = 0 ; i<row ;i++){
        for(int j = 0 ; j<col ; j++){
            h_b[i*col + j] = b[i][j];
        }
    }   


    cudaMemcpy(d_a, h_a, row * col * sizeof(float),cudaMemcpyHostToDevice );
    cudaMemcpy(d_b, h_b, row * col * sizeof(float),cudaMemcpyHostToDevice );

    dim3 blockSize(16, 16);
    dim3 gridSize((col + blockSize.x - 1) / blockSize.x, (row + blockSize.y - 1) / blockSize.y);
    addMatrix<<<gridSize,blockSize>>>(row, col, d_a, d_b);
    cudaMemcpy(h_ans, d_a  , row * col * sizeof(float), cudaMemcpyDeviceToHost);
       for(int i = 0 ; i<row ; i++){
        for(int j = 0 ; j<col ; j++){
            a[i][j] = h_ans[i*col + j];
        }
      }   

    free(h_ans);
    free(h_a);
    free(h_b);

    cudaFree(d_a);
    cudaFree(d_b);
}


void cudaReLU(float** matrix, int row, int col){

    float* d_matrix;
    float* h_matrix;
    float* h_ans;
    cudaMalloc((void**)&d_matrix, row * col * sizeof(float));
    h_matrix = (float*) malloc(row * col* sizeof(float));   
    h_ans = (float*) malloc(row * col* sizeof(float));   

    for(int i = 0 ; i<row ;i++){
        for(int j = 0 ; j<col ; j++){
            h_matrix[i*col + j] = matrix[i][j];
        }
    }    
    cudaMemcpy(d_matrix, h_matrix, row * col * sizeof(float),cudaMemcpyHostToDevice ); 


    dim3 blockSize(16, 16);
    dim3 gridSize((col + blockSize.x - 1) / blockSize.x, (row + blockSize.y - 1) / blockSize.y);
    ReLU<<<gridSize,blockSize>>>(row, col, d_matrix);
    cudaMemcpy(h_ans, d_matrix  , row * col * sizeof(float), cudaMemcpyDeviceToHost);
       for(int i = 0 ; i<row ; i++){
        for(int j = 0 ; j<col ; j++){
            matrix[i][j] = h_ans[i*col + j];
        }
      }   

    free(h_ans);
    free(h_matrix);


    cudaFree(d_matrix);
}

void cudaTanH(float** matrix, int row, int col){

    float* d_matrix;
    float* h_matrix;
    float* h_ans;
    cudaMalloc((void**)&d_matrix, row * col * sizeof(float));
    h_matrix = (float*) malloc(row * col* sizeof(float));   
    h_ans = (float*) malloc(row * col* sizeof(float));   

    for(int i = 0 ; i<row ;i++){
        for(int j = 0 ; j<col ; j++){
            h_matrix[i*col + j] = matrix[i][j];
        }
    }    
    cudaMemcpy(d_matrix, h_matrix, row * col * sizeof(float),cudaMemcpyHostToDevice ); 


    dim3 blockSize(16, 16);
    dim3 gridSize((col + blockSize.x - 1) / blockSize.x, (row + blockSize.y - 1) / blockSize.y);
    tanH<<<gridSize,blockSize>>>(row, col, d_matrix);
    cudaMemcpy(h_ans, d_matrix  , row * col * sizeof(float), cudaMemcpyDeviceToHost);
       for(int i = 0 ; i<row ; i++){
        for(int j = 0 ; j<col ; j++){
            matrix[i][j] = h_ans[i*col + j];
        }
      }   

    free(h_ans);
    free(h_matrix);


    cudaFree(d_matrix);
}

void softmax(int n, float* vec){
    float dr = 0;
    for(int i = 0 ; i<n ; i++){
        dr += (float)exp(vec[i]);
    }
    for(int i = 0 ; i<n ; i++){
        vec[i] = (float)exp(vec[i])/dr;
    }
}

void freeMatrix(float** matrix,int row){
    for(int i = 0 ; i<row ; i++){
        free(matrix[i]);
    }
    free(matrix);
}

float** createMatrix(int row , int col){

    float** matrix = (float**) malloc(sizeof(float*) * row);
    for(int i = 0 ; i<row; i++){
        matrix[i] = (float*) malloc(sizeof(float) * col);
    }

    return matrix;
}

void sigmoid(int n, float* vec){
    for(int i = 0 ; i<n ; i++){
        vec[i] = 1/(1+exp(-vec[i]));
    }
}

float** createPaddedMatrix(int originalRows, int originalCols, float** originalMatrix, int padding) {
    int paddedRows = originalRows + 2 * padding;
    int paddedCols = originalCols + 2 * padding;

    
    float** paddedMatrix = (float**) malloc(paddedRows * sizeof(float*));
    for(int i = 0; i < paddedRows; i++) {
        paddedMatrix[i] = (float*) calloc(paddedCols, sizeof(float)); // Using calloc to initialize to 0
    }

    
    for(int i = 0; i < originalRows; i++) {
        for(int j = 0; j < originalCols; j++) {
            paddedMatrix[i + padding][j + padding] = originalMatrix[i][j];
        }
    }

    return paddedMatrix;
}

void convolution_With_Padding(int Padding_factor , int mRow, int mCol, int knRow, int knCol, float** kernel, float** matrix, float** ans, float bias){


    float** paddedMatrix = createPaddedMatrix(mRow,mCol,matrix,Padding_factor);
    int paddedRows = mRow + 2 * Padding_factor;
    int paddedCols = mCol + 2 * Padding_factor;
    cudaConvolution(paddedRows,paddedCols,knRow,knCol,kernel,paddedMatrix,ans,bias);
    
    for(int i = 0 ; i<paddedRows ; i++){
        free(paddedMatrix[i]);
    }
    free(paddedMatrix);


}



int main(int argc,char** argv){
    int action = atoi(argv[1]);

    if(action==1){
        int order_matrix = atoi(argv[2]);
        int order_kernel = atoi(argv[3]);
        int Padding_factor = atoi(argv[4]);
        int order_outputMatrix = order_matrix - order_kernel + 2*Padding_factor + 1;
        float** original_matrix = createMatrix(order_matrix, order_matrix);
        float** kernel = createMatrix(order_kernel,order_kernel);
        float** ans = createMatrix(order_outputMatrix,order_outputMatrix);
        for(int i = 0 ; i<order_matrix; i++){
            for(int j = 0 ; j<order_matrix ; j++){
                original_matrix[i][j] = atoi(argv[4 + i*order_matrix + j]);
            }
        }

        for(int i = 0 ; i<order_kernel; i++){
            for(int j = 0 ; j<order_kernel ; j++){
                kernel[i][j] = atoi(argv[5 + order_matrix*order_matrix + i*order_kernel + j]);
            }
        }

        // float** Padded_matrix = createPaddedMatrix(order_matrix, order_matrix, original_matrix, Padding_factor);
        convolution_With_Padding(Padding_factor, order_matrix, order_matrix, order_kernel, order_kernel, kernel, original_matrix, ans, 0);
        printf("\n\n");

        for(int i = 0 ; i < order_outputMatrix; i++){
            for(int j = 0 ; j<order_outputMatrix ; j++){
                printf("%f ",ans[i][j] );
            }
            printf("\n");
        }


    }

    else if(action == 2){
        int activation_choice = atoi(argv[2]);
        int order_matrix_row = atoi(argv[3]);
        int order_matrix_col = atoi(argv[4]);
        float** matrix = createMatrix(order_matrix_row, order_matrix_col);
        for(int i = 0 ; i<order_matrix_row ; i++){
            for(int j = 0 ; j<order_matrix_col ; j++){
                matrix[i][j] = atoi(argv[5 + i*order_matrix_col + j]);
            }
        }
        if(activation_choice==0){
                    cudaReLU(matrix, order_matrix_row, order_matrix_col);
        }
        else if(activation_choice==1){
            cudaTanH(matrix, order_matrix_row, order_matrix_col);
        }
        printf("\n\n");
        for(int i = 0 ; i<order_matrix_row ; i++){
            for(int j = 0 ; j<order_matrix_col ; j++){
                printf("%f ",matrix[i][j]);
            }
            printf("\n");
        }   



    }

    else if(action == 3){

        int pooling_choice = atoi(argv[2]);
        int pooling_size = atoi(argv[3]);
        int order_matrix = atoi(argv[4]);
        float** matrix = createMatrix(order_matrix, order_matrix);
        for(int i = 0 ; i<order_matrix ; i++){
            for(int j = 0 ; j<order_matrix ; j++){
                matrix[i][j] = atoi(argv[5 + i*order_matrix + j]);
            }
        }
        float** ans = createMatrix(order_matrix/pooling_size,order_matrix/pooling_size);
        if(pooling_choice == 0){
            cudaMaxPooling(order_matrix, order_matrix, pooling_size, matrix, ans);
        }
        else if(pooling_choice == 1){
            cudaAvgPooling(order_matrix, order_matrix, pooling_size, matrix, ans);
        }
        printf("\n\n");
        for(int i = 0 ; i< order_matrix/pooling_size ; i++){
            for(int j = 0 ; j<order_matrix/pooling_size ; j++){
                printf("%f ",ans[i][j]);
            }
            printf("\n");
        }

    }
    else if(action == 4){
        int prob = atoi(argv[2]);
        float vec[10];
        for(int i = 0 ; i<10 ; i++){
            vec[i] = atoi(argv[3+i]);
        }
        if(prob == 0){
            sigmoid(10, vec);
        }
        else if(prob == 1){
            softmax(10, vec);
        }
        printf("\n\n");
        for(int i = 0 ; i<10 ; i++){
            printf("%f ",vec[i]);
        }
        printf("\n");

    }

}
