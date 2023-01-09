#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define w 400
using namespace cv;

double h = 1e-4;
double learning_rate = 1e-2;
int batch_size = 1;
const int n =4000;

double b_1 = 0.9;
double b_2 = 0.999;

const int size_of_X = 1;

double* X = (double*)malloc(sizeof(double) * size_of_X);
double* X2 = (double*)malloc(sizeof(double) * size_of_X);
double* Y = (double*)malloc(sizeof(double) * size_of_X);


double* A = (double*)malloc(sizeof(double) * n);
double* mt_1 = (double*)malloc(sizeof(double) * n);
double* vt_1 = (double*)malloc(sizeof(double) * n);
double* mt = (double*)malloc(sizeof(double) * n);
double* vt = (double*)malloc(sizeof(double) * n);
double* _m = (double*)malloc(sizeof(double) * n);
double* _v = (double*)malloc(sizeof(double) * n);
double* DROPOUT = (double*)malloc(sizeof(double) * n);


const int img_w = 5 * 5;
double x_input[img_w] = { 0 };


double f(double x) {
	//printf("%f  ", 62 * pow(x, 2) - x + 99);
	return -18 * pow(x, 5) + 50 * pow(x, 4) - x + 30;
}

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double relu(double x) {
	if (x > 0)return x;
	if (x <= 0)return 0;
}

double tanh(double x) {
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double F(double AK[], double x_input[]) {
	/*double a0 = sigmoid(AK[0] * x_input[0] + AK[1]);
	double a1 = sigmoid(AK[2] * x_input[1] + AK[3]);
	double a2 = sigmoid(AK[4] * a0 + AK[5] * a1 + AK[6]);
	*/
	int i, j;
	double a_input[img_w];
	for (i = 0; i < img_w; i++) {
		a_input[i] = relu(AK[i * 2 + 0] * x_input[i] + AK[i * 2 + 1]);

	}
	int temp = i*2+1;


	const int hidden_0 = 25;
	double a_hidden_0[hidden_0];
	for (i = 0; i < hidden_0; i++) {
		double x_sum = 0;
		for (j = 0; j < img_w; j++) {
			x_sum += AK[temp + j] * a_input[j];
		}
		x_sum += AK[temp +  j];
		a_hidden_0[i] = relu(x_sum);

		temp = temp + img_w + 2;
	}

	const int hidden_1 =20;
	double a_hidden_1[hidden_1];
	for (i = 0; i < hidden_1; i++) {
		double x_sum = 0;
		for (j = 0; j < hidden_0; j++) {
			x_sum += AK[temp + j] * a_hidden_0[j];
		}
		x_sum += AK[temp +  j];
		a_hidden_1[i] = relu(x_sum);
		temp = temp + hidden_0 + 2;
	}
	
	
	const int hidden_2 = 20;
	double a_hidden_2[hidden_2];
	for (i = 0; i < hidden_2; i++) {
		double x_sum = 0;
		for (j = 0; j < hidden_1; j++) {
			x_sum += AK[temp + j] * a_hidden_1[j];
		}
		x_sum += AK[temp +  j];
		a_hidden_2[i] = relu(x_sum);
		temp = temp +  hidden_1 + 2;
	}

	const int hidden_3 = 10;
	double a_hidden_3[hidden_3];
	for (i = 0; i < hidden_3; i++) {
		double x_sum = 0;
		for (j = 0; j < hidden_2; j++) {
			x_sum += AK[temp + j] * a_hidden_2[j];
		}
		x_sum += AK[temp + j];
		a_hidden_3[i] = relu(x_sum);
		temp = temp + hidden_2 + 2;
	}

	const int hidden_4 = 10;
	double a_hidden_4[hidden_4];
	for (i = 0; i < hidden_4; i++) {
		double x_sum = 0;
		for (j = 0; j < hidden_3; j++) {
			x_sum += AK[temp + j] * a_hidden_3[j];
		}
		x_sum += AK[temp + j];
		a_hidden_4[i] = relu(x_sum);
		temp = temp + hidden_3 + 2;
	}

	const int hidden_5 = 10;
	double a_hidden_5[hidden_5];
	for (i = 0; i < hidden_5; i++) {
		double x_sum = 0;
		for (j = 0; j < hidden_4; j++) {
			x_sum += AK[temp + j] * a_hidden_4[j];
		}
		x_sum += AK[temp + j];
		a_hidden_5[i] = relu(x_sum);
		temp = temp + hidden_4 + 2;
	}
	/*
	const int hidden_6 = 20;
	double a_hidden_6[hidden_6];
	for (i = 0; i < hidden_6; i++) {
		double x_sum = 0;
		for (j = 0; j < hidden_5; j++) {
			x_sum += AK[temp + j] * a_hidden_5[j];
		}
		x_sum += AK[temp + j];
		a_hidden_6[i] = relu(x_sum);
		temp = temp + hidden_5 + 2;
	}*/


	const int output = 1;
	double a_output[output];
	double x_sum = 0;
	for (j = 0; j < hidden_5; j++) {
		x_sum += AK[temp + j] * a_hidden_5[j];
	}
	x_sum += AK[temp + j ];

	a_output[0] = tanh(x_sum);
	//printf("%f \n", a_output[0]);

	//double a4 = sigmoid(AK[10] * a2 + AK[11] * a3 + AK[12]);
	/*
	double a0 = sigmoid(AK[0] * x_input[0] + AK[1]);
	double a1 = sigmoid(AK[2] * x_input[1]+AK[3]);
	double a2 = sigmoid(AK[4] * a0 + AK[5] * a1 + AK[6]);
	double a3 = sigmoid(AK[7] * a0 + AK[8] * a1 + AK[9]);
	double a4 = sigmoid(AK[10] * a2 + AK[11] * a3 + AK[12]);
	*/
	/*
	double y = 0;
	int i;
	for (i = 1; i < n + 1; i++) {
		y += AK[i - 1] * pow(x, n - i);
		//printf("!!%d!!", AK[i-1]);
	}
	*/

	return a_output[0];
}

double partialW(double A[], double  x_input[], double y, int N) {

	double A1[n] = { 0 };
	double A2[n] = { 0 };
	int i;
	for (i = 0; i < n; i++) {
		A1[i] = A[i];
		A2[i] = A[i];

	}
	A1[N] += h;

	//printf("\n");
	//printf("<%f--%f--%f>\n   ",  F(A1, x_input), pow(y - F(A1, x_input), 2), (pow(y - F(A1, x_input), 2) - pow(y - F(A2, x_input), 2)) / h);
	return (pow(y - F(A1, x_input), 2) - pow(y - F(A2, x_input), 2)) / h;


}



double train(double A[], double x_input[], double y) {
	int i, j;
	for (j = 0; j < n; j++) {

		if (DROPOUT[j] == 0){

		int arr[size_of_X] = { 0 };
		//printf("%f", size_of_X);
		for (i = 0; i < size_of_X; i++) {
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


			//x_input[0] = x;
			//x_input[1] = x2;

			double g = partialW(A, x_input, y, j);
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
	}
	return 0;

}
void main() {
	srand(time(NULL));


	//Mat atom_image = Mat::zeros(w, w, CV_8UC3);

	/*Point p2;
	p2.x = 100;
	p2.y = 100;
	//画实心点
	circle(atom_image, p2, 3, Scalar(255, 0, 0), -1);
	*/
	int i, j;

	for (i = 0; i < n; i++) {
		A[i] = rand() % 100 / 100.0 - 0.5;
		mt_1[i] = 0;
		vt_1[i] = 0;
		mt[i] = 0;
		vt[i] = 0;
		_m[i] = 0;
		_v[i] = 0;
		DROPOUT[i] = 0;
	}

	/*
	for (i = 0; i < size_of_X; i++) {
		X[i] = i / 100.0 - 1;

		if (i > size_of_X / 2) { Y[i]=1; }
		else { Y[i]=0; }
	}*/

	double error = 0;
	double lerr = 999;

	for (i = 0; i < 50; i++) {
		printf("!");

		for(j=0;j<n;j++)DROPOUT[j] = 0;


		for (j = 0; j < n * 0.3; j++) {
			int index = rand() % n;
			while (DROPOUT[index] == 1) {
				index = rand() % n;
			}
			DROPOUT[index] = 1;
		}
		int rrr = 0;
		error = 0;
		for (rrr = 0; rrr < 5; rrr++) {



			int kkk = 0;
			char str[] = "C:\\Users\\b4100\\Desktop\\Gradient descent C++\\5\\";
			String str2 = std::to_string(rand() % 5);
			char str3[] = ".png";
			String Str;
			Str.append(str);
			Str.append(str2);
			Str.append(str3);

			//printf("%s", Str);

			Mat img2 = imread(Str, 0);
			Mat img;
			resize(img2, img, Size(5, 5), INTER_LINEAR);

			int i, j;
			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img.at<uchar>(i, j) / 255.0;

				}
			}



			imshow("test", img2);
			waitKey(1);

			//printf("%f", A[2]);


			train(A, x_input, 0);

			char strA[] = "C:\\Users\\b4100\\Desktop\\Gradient descent C++\\8\\";
			String str2A = std::to_string(rand() % 5);
			char str3A[] = ".png";
			String StrA;
			StrA.append(strA);
			StrA.append(str2A);
			StrA.append(str3A);

			//printf("%s", StrA);

			Mat img3 = imread(StrA, 0);
			Mat img4;
			resize(img3, img4, Size(5, 5), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img4.at<uchar>(i, j) / 255.0;

				}
			}



			imshow("test", img3);
			waitKey(1);
			//destroyAllWindows();
			train(A, x_input, 1);
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


			


			char strB[] = "C:\\Users\\b4100\\Desktop\\Gradient descent C++\\5\\";
			String str2B = std::to_string(rand() % 5);
			char str3B[] = ".png";
			String StrB;
			StrB.append(strB);
			StrB.append(str2B);
			StrB.append(str3B);

			//printf("%s", StrB);

			Mat img5 = imread(StrB, 0);
			Mat img6;
			resize(img5, img6, Size(5, 5), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img6.at<uchar>(i, j) / 255.0;

				}
			}
			error += pow(0 - F(A, x_input), 2);


			//imshow("test", img2);
			//waitKey(1);



			train(A, x_input, 0);


			char strC[] = "C:\\Users\\b4100\\Desktop\\Gradient descent C++\\8\\";
			String str2C = std::to_string(rand() % 5);
			char str3C[] = ".png";
			String StrC;
			StrC.append(strC);
			StrC.append(str2C);
			StrC.append(str3C);

			//printf("%s", StrC);

			Mat img7 = imread(StrC, 0);
			Mat img8;
			resize(img7, img8, Size(5, 5), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img8.at<uchar>(i, j) / 255.0;

				}
			}

			error += pow(1 - F(A, x_input), 2);

		}
		printf("loss = %f \n", error);

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

	/*
	printf("\n"); printf("\n"); printf("\n");
	for (j = 0; j < n; j++) {
		printf("%f  ", A[j]);
	}
	printf("\n"); printf("\n"); printf("\n");
	*/

	while (true)
	{
		int input;
		printf("輸入圖片(5是五，8是八):");
		scanf("%d", &input);
		if (input == 8) {

			char strC[] = "C:\\Users\\b4100\\Desktop\\Gradient descent C++\\8\\";
			String str2C = std::to_string(rand() % 5);
			char str3C[] = ".png";
			String StrC;
			StrC.append(strC);
			StrC.append(str2C);
			StrC.append(str3C);

			//printf("%s", StrC);

			Mat img9 = imread(StrC, 0);

			Mat img10;
			resize(img9, img10, Size(5, 5), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img10.at<uchar>(i, j) / 255.0;

				}
			}
			int txt = 8;
			if (F(A, x_input) < 0.5)txt = 5;
			printf("預測數值=%f  預測類別=%d\n\n", F(A, x_input), txt);
			imshow("test", img9);
			waitKey(1);
		}
		if (input == 5) {

			char strC[] = "C:\\Users\\b4100\\Desktop\\Gradient descent C++\\5\\";
			String str2C = std::to_string(rand() % 5);
			char str3C[] = ".png";
			String StrC;
			StrC.append(strC);
			StrC.append(str2C);
			StrC.append(str3C);

			//printf("%s", StrC);

			Mat img11 = imread(StrC, 0);

			Mat img12;
			resize(img11, img12, Size(5, 5), INTER_LINEAR);


			for (i = 0; i < 5; i++) {
				for (j = 0; j < 5; j++) {
					x_input[i * 5 + j] = img12.at<uchar>(i, j) / 255.0;

				}
			}
			int txt = 8;
			if (F(A, x_input) < 0.5)txt = 5;
			printf("預測數值=%f  預測類別=%d\n\n", F(A, x_input), txt);
			imshow("test", img11);
			waitKey(1);
		}

	}

	//waitKey(0);
	//destroyAllWindows();
	destroyAllWindows();
	system("pause");

}
