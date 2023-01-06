#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define w 400
using namespace cv;

double h = 1e-8;
double learning_rate = 1e-3;
int batch_size = 2;
const int n = 13;

double b_1 = 0.9;
double b_2 = 0.999;

const int size_of_X = 4;
double X[size_of_X] =   { 1,1,0,0};
double X2[size_of_X] = { 1,0,1,0 };
double Y[size_of_X] =   { 1,0,0,1 };

double A[n] = { 0 };
double mt_1[n] = { 0 };
double vt_1[n] = { 0 };
double mt[n] = { 0 };
double vt[n] = { 0 };
double _m[n] = { 0 };
double _v[n] = { 0 };



double x_input[2] = { 0,0 };


double f(double x) {
	//printf("%f  ", 62 * pow(x, 2) - x + 99);
	return - 18 * pow(x, 5) + 50 * pow(x, 4) - x + 30;
}

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double F(double AK[], double x_input[]) {
	/*double a0 = sigmoid(AK[0] * x_input[0] + AK[1]);
	double a1 = sigmoid(AK[2] * x_input[1] + AK[3]);
	double a2 = sigmoid(AK[4] * a0 + AK[5] * a1 + AK[6]);
	*/
	
	double a0 = sigmoid(AK[0] * x_input[0]+AK[1]);
	double a1 = sigmoid(AK[2] * x_input[1]+AK[3]);
	double a2 = sigmoid(AK[4] * a0 + AK[5] * a1 + AK[6]);
	double a3 = sigmoid(AK[7] * a0 + AK[8] * a1 + AK[9]);
	double a4 = sigmoid(AK[10] * a2 + AK[11] * a3 + AK[12]);
	
	/*
	double y = 0;
	int i;
	for (i = 1; i < n + 1; i++) {
		y += AK[i - 1] * pow(x, n - i);
		//printf("!!%d!!", AK[i-1]);
	}

	*/

	return a4;
}

double partialW(double A[],double  x_input[], double y, int N) {

	double A1[n] = { 0 };
	double A2[n] = { 0 };
	int i;
	for (i = 0; i < n; i++) {
		A1[i] = A[i];
		A2[i] = A[i];

	}
	A1[N] += h;

	//printf("\n");
	//printf("<%f--%f--%f>   ",  F(A1, x), pow(y - F(A1, x), 2), (pow(y - F(A1, x), 2) - pow(y - F(A2, x), 2)) / h);
	return (pow(y - F(A1, x_input), 2) - pow(y - F(A2, x_input), 2)) / h;


}



double train(double A[],double X[], double Y[]) {
	int i, j;
	for (j = 0; j < n; j++) {
		int arr[size_of_X] = { 0 };
		//printf("%f", size_of_X);
		for (i = 0; i< size_of_X; i++) {
			arr[i] = 0;
		}
		
		for (i = 0; i < batch_size; i++) {
			int index = rand() % size_of_X;
			while (arr[index] == 1) {
				index = rand() % size_of_X;
				
			}
			//printf("eWegerh");
			//printf("%d",index);

			arr[index] = 1;
			double x = X[index];
			double x2 = X2[index];
			double y = Y[index];
			x_input[0] = x;
			x_input[1] = x2;

			double g = partialW(A,x_input, y, j);
			//printf("<%f>", g);

			mt[j] = b_1 * mt_1[j] + (1 - b_1) * g;
			vt[j] = b_2 * vt_1[j] + (1 - b_2) * pow(g, 2);
			_m[j] = mt[j] / (1 - b_1);
			_v[j] = vt[j] / (1 - b_2);

			vt_1[j] = vt[j];
			mt_1[j] = mt[j];

			A[j] = A[j] - learning_rate * _m[j] / (sqrt(_v[j]) + 1e-8);

		}
	}
	return 0;

}

void main() {

	//Mat atom_image = Mat::zeros(w, w, CV_8UC3);
	
	/*Point p2;
	p2.x = 100;
	p2.y = 100;
	//画实心点
	circle(atom_image, p2, 3, Scalar(255, 0, 0), -1);
	*/
	int i, j;

	for (i = 0; i < n; i++) {
		A[i] = rand()%2-1;
		mt_1[i] = 0;
		vt_1[i] = 0;
		mt[i] = 0;
		vt[i] = 0;
		_m[i] = 0;
		_v[i] = 0;
	}

	/*
	for (i = 0; i < size_of_X; i++) {
		X[i] = i / 100.0 - 1;
		
		if (i > size_of_X / 2) { Y[i]=1; }
		else { Y[i]=0; }


	}*/

	double error = 0;
	double lerr = 999;

	for (i = 0; i < 500000; i++) {
		printf("!");


		train(A, X,Y);
		
		/*
		if(i%10==0){
			for (j = 0; j < 200; j++) {
				Point p1;
				p1.x = j*2;
				p1.y =0.5*w-f(j/100.0-1);
				//画实心点5
				circle(atom_image, p1, 1, Scalar(255,255, 255), -1);


				Point p2;
				p2.x = j * 2;
				x_input[0] = j / 100.0 - 1;
				x_input[1] = 1.3*j / 100.0 - 1;
				p2.y = 0.5*w-F(A, x_input);
				//画实心点
				circle(atom_image, p2, 1, Scalar(255, 0, 255), -1);
			}
		


			//imshow("test", atom_image);
			//waitKey(1);
		}*/


		error = 0;

		for (j = 0; j< size_of_X; j++) {
			//double x = X[j];
			x_input[0] = X[j];
			x_input[1] = X2[j];
			double y = Y[j];
			error += pow(y - F(A, x_input), 2);

			
		}


	 printf("%f \n", error);
		
		/*for (j = 0; j < n; j++) {
			printf("%f ", A[j]);
		}
		printf("\n");*/

		//if (fabs(lerr - error) < 1e-9) {
		//	break;
		//	printf("!!!!!!!!!!!");
		//}



		lerr = error;

	}


	printf("\n"); printf("\n"); printf("\n");
	for (j = 0; j < n; j++) {
		printf("%f  ", A[j]);
	}
	printf("\n"); printf("\n"); printf("\n");


	//waitKey(0);
	//destroyAllWindows();

	system("pause");

}
