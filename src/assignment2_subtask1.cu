#include<stdio.h>
#include <stdlib.h>
#include<math.h>


void ReLU4D(int num_inputChannel, int order_inputMatrix,float* inputLayer){
	
	for(int i = 0 ; i<num_inputChannel ; i++){


		for(int x = 0 ; x<order_inputMatrix ; x++){
			for(int y = 0 ; y<order_inputMatrix ; y++){
				if(inputLayer[i*order_inputMatrix*order_inputMatrix + x*order_inputMatrix + y] < 0) inputLayer[i*order_inputMatrix*order_inputMatrix + x*order_inputMatrix + y] = 0;
			}
		}


	}

}

void ReLU(float** matrix, int row, int col){
		for(int i = 0 ; i<row ; i++){
			for(int j = 0 ; j<col ; j++){
				if(matrix[i][j]<0) matrix[i][j] = 0;
			}
		}
}

void tanH(float** matrix, int row, int col){

		for(int i = 0 ; i<row ; i++){
			for(int j = 0 ; j<col ; j++){
				float temp = matrix[i][j];
				temp = (exp(temp)-exp(-temp))/(exp(temp)+exp(-temp));
				matrix[i][j] = temp;
			}
		}


}



void addMatrix(int row, int col ,float** a , float** b ){
	

	for(int i = 0 ; i<row ; i++){
		for(int j = 0 ; j<col ; j++){
			a[i][j] = a[i][j] + b[i][j];
		}
	}
}


void convolution4D(int num_inputChannel, int num_outputChannel, int order_inputMatrix, int order_kernel, float* inputLayer, float* weights,float* outputLayer){
	int order_outputMatrix = order_inputMatrix - order_kernel + 1;



	for(int i = 0 ; i<num_outputChannel ; i++){
		for(int j = 0 ; j<num_inputChannel ; j++){




				for(int x = 0 ; x<order_outputMatrix ; x++){
					for(int y = 0 ; y<order_outputMatrix ; y++){
						float sum = 0;
						for(int p = 0 ; p<order_kernel ; p++){
							for(int k = 0 ; k<order_kernel ; k++){
								sum += weights[ i*num_inputChannel*order_kernel*order_kernel + j*(order_kernel*order_kernel) + p*order_kernel + k] * inputLayer[j*order_inputMatrix*order_inputMatrix + (x+p)*order_inputMatrix + (y+k)];
							}
						}
						outputLayer[i*order_outputMatrix*order_outputMatrix + x*order_outputMatrix + y] += sum;
					}
				}




		}
	}

}



void maxPooling4D(int num_inputChannel, int order_inputMatrix,float* inputLayer,float* outputLayer){
	int order_outputMatrix = order_inputMatrix/2;

	for(int i = 0 ; i<num_inputChannel ; i++){


		for(int x = 0 ; x<order_inputMatrix ; x+=2){
			for(int y = 0 ; y<order_inputMatrix ; y+=2){
				float maxi =inputLayer[i*order_inputMatrix*order_inputMatrix + x*order_inputMatrix + y];
				for(int p = x ; p<x+2 ; p++){
					for(int k = y ; k<y+2 ; k++){
						if(maxi < inputLayer[i*order_inputMatrix*order_inputMatrix + p*order_inputMatrix + k]) maxi = inputLayer[i*order_inputMatrix*order_inputMatrix + p*order_inputMatrix + k];
					}
				}
				outputLayer[i*order_outputMatrix*order_outputMatrix + (x/2)*order_outputMatrix + (y/2)] = maxi;
			}
		}


	}


}



//called in main.c
void convolution(int mRow, int mCol, int knRow, int knCol,float** kernel, float** matrix, float** ans, float bias ){
	int ansRow = mRow - knRow + 1;
	int ansCol = mCol - knCol + 1;

	for(int i = 0 ; i<ansRow ; i++){
		for(int j = 0 ; j<ansCol ; j++){
			float sum = 0;
			for(int p = 0 ; p<knRow ; p++){
				for(int k = 0 ; k<knCol ; k++){
					sum += kernel[p][k] * matrix[i+p][j+k];
				}
			}
			ans[i][j] = sum + bias;
		}
	}

}


//called in main.c
void maxPooling(int mRow, int mCol, int pooling_size,float** matrix, float** ans){
	for(int i = 0 ; i<mRow ; i+=pooling_size){
		for(int j = 0 ; j<mCol ; j+=pooling_size){
			float maxi = matrix[i][j];
			for(int p = i ; p<i+pooling_size ; p++){
				for(int k = j ; k<j+pooling_size ; k++){
					if(maxi < matrix[p][k]) maxi = matrix[p][k];
				}
			}
			ans[i/pooling_size][j/pooling_size] = maxi;
		}
	}


}

void avgPooling(int mRow, int mCol, int pooling_size, float** matrix, float** ans){

	int ansRow = mRow/pooling_size;
	int ansCol = mCol/pooling_size;

	for(int i = 0 ; i<mRow ; i+=pooling_size){
		for(int j = 0 ; j<mCol ; j+=pooling_size){
			float sum = 0;
			for(int p = i ; p<i+pooling_size ; p++){
				for(int k = j ; k<j+pooling_size ; k++){
					sum += matrix[p][k];
				}
			}
			ans[i/pooling_size][j/pooling_size] = sum/(pooling_size*pooling_size);
		}
	}



}

void sigmoid(int n, float* vec){
	for(int i = 0 ; i<n ; i++){
		vec[i] = 1/(1+exp(-vec[i]));
	}
}



//called in main.c
void softmax(int n, float* vec){
	float dr = 0;
	for(int i = 0 ; i<n ; i++){
		dr += (float)exp(vec[i]);
		vec[i] = (float)exp(vec[i]);
	}
	for(int i = 0 ; i<n ; i++){
		vec[i] = vec[i]/dr;
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
	convolution(paddedRows,paddedCols,knRow,knCol,kernel,paddedMatrix,ans,bias);
	
	for(int i = 0 ; i<paddedRows ; i++){
		free(paddedMatrix[i]);
	}
	free(paddedMatrix);


}



//called in main.c
float** createMatrix(int row , int col){

	float** matrix = (float**) malloc(sizeof(float*) * row);
	for(int i = 0 ; i<row; i++){
		matrix[i] = (float*) malloc(sizeof(float) * col);
	}

	return matrix;
}



//called in main.c
void freeMatrix(float** matrix,int row){
	for(int i = 0 ; i<row ; i++){
		free(matrix[i]);
	}
	free(matrix);
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
					ReLU(matrix, order_matrix_row, order_matrix_col);
		}
		else if(activation_choice==1){
			tanH(matrix, order_matrix_row, order_matrix_col);
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
			maxPooling(order_matrix, order_matrix, pooling_size, matrix, ans);
		}
		else if(pooling_choice == 1){
			avgPooling(order_matrix, order_matrix, pooling_size, matrix, ans);
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
