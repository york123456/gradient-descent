#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double h = 1e-8;
double learning_rate = 1e-2;
int batch_size = 5;
const int n = 6;

double b_1 = 0.9;
double b_2 = 0.999;

double X[200] = { 0 };
double Y[200] = { 0 };

double A[n] = { 0 };
double mt_1[n] = { 0 };
double vt_1[n] = { 0 };
double mt[n] = { 0 };
double vt[n] = { 0 };
double _m[n] = { 0 };
double _v[n] = { 0 };


double f(double x) {
	//printf("%f  ", 62 * pow(x, 2) - x + 99);
	return  1 * pow(x, 5) +99*pow(x,4) - x + 7;
}

double F(double AK[], double x) {
	double y = 0;
	int i;
	for (i = 1; i <n+1; i++) {
		y += AK[i-1] * pow(x, n - i);
		//printf("!!%d!!", AK[i-1]);
	}
	
	return y;
}

double partialW(double  x, double y, int N) {

	double A1[n] = { 0 };
	double A2[n] = { 0 };
	int i;
	for (i = 0; i <n; i++) {
		A1[i] = A[i];
		A2[i] = A[i];
		
	}
	A1[N] += h;
	
	//printf("\n");
	//printf("<%f--%f--%f>   ",  F(A1, x), pow(y - F(A1, x), 2), (pow(y - F(A1, x), 2) - pow(y - F(A2, x), 2)) / h);
	return (pow(y - F(A1, x), 2) - pow(y - F(A2, x), 2)) / h;


}



double train(void) {
	int i, j;
	for (j = 0; j<n; j++) {
		int arr[int(sizeof(X) / sizeof(double))] = {0};

		for (i = 0; i<int(sizeof(X) / sizeof(double)); i++) {
			arr[i] = 0;
		}

		for (i = 0; i < batch_size; i++) {
			int index = rand() % int(sizeof(X) / sizeof(double));
			while (arr[index] == 1) {
				index = rand() % int(sizeof(X) / sizeof(double));
			}
		
			arr[index] = 1;
			double x = X[index];
			double y = Y[index];
			double g = partialW( x, y, j);
			//printf("<%f>", g);

			mt[j] = b_1 * mt_1[j] + (1 - b_1) * g;
			vt[j] = b_2 * vt_1[j] + (1 - b_2) * pow(g, 2);
			_m[j] = mt[j] / (1 - b_1);
			_v[j] = vt[j] / (1 - b_2);

			vt_1[j] = vt[j];
			mt_1[j] = mt[j];
			
			A[j] = A[j] - learning_rate * _m[j] / (sqrt(_v[j] ) + 1e-8);

		}
	}
	return 0;

}

void main() {
	

	int i,j;

	for (i = 0; i < n; i++) {
		A[i] = 0;
		mt_1[i] =  0 ;
		vt_1[i] = 0;
		mt[i] = 0;
		vt[i] = 0;
		_m[i] = 0;
		_v[i] = 0;
	}

	for (i = 0; i < int(sizeof(X) / sizeof(double)); i++) {
		X[i] = i / 100.0 - 1;
		Y[i] = f(X[i]);

		

	}

	double error = 0;
	double lerr = 999;

	for (i = 0; i <500000; i++) {
		train();
		error = 0;

		for (j = 0; j< int(sizeof(X) / sizeof(double)); j++) {
			double x = X[j];
			double y = Y[j];
			error += pow(y - F(A, x), 2);

			

		}
		
		
		//printf("%f -", error);
		for (j = 0; j<n; j++) {
			printf("%f ", A[j]);
		}
		printf("\n");

		if (fabs(lerr - error )< 1e-9) {
			break;
			printf("!!!!!!!!!!!");
		}



		lerr = error;

	}


	printf("\n"); printf("\n"); printf("\n");
	for (j = 0; j < n; j++) {
		printf("%f*x^%d+ ", A[j],n-j-1);
	}
	printf("\n"); printf("\n"); printf("\n");


	system("pause");

}
