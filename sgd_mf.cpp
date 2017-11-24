#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thread>
#include <random>
#include <map>
#include "lda.h"

#define PROJECT "VV"

using namespace std;

#define FEATURE_OFFSET 3
#define USER_BIAS_INDEX 1
#define ITEM_BIAS_INDEX 2

#define INDEX(i, j) (i * rank + j)
#define NOISE 0.02

struct LogData {
	int item_id;
	int user_id;
	double value;
};

class SgdMf {
	int n_iter;
	int rank;
	double lambda;
	double biasMuRatio = 0.5;
	double biasLambdaRatio=0.1;
	double mu0 = 0.01;
	double decayFactor = 1;
	int stepOffset = 0;
	double forgettingExponent = 0;
	double *userVectors;
	double *itemVectors;
	
	void initialize(LogData *sample, double *userVectors, double *itemVectors, int n_users, int n_items) {
		random_device seed_gen;
		default_random_engine engine(seed_gen());
		normal_distribution<> dist(0.0, 1.0);
		
	    double mean_value = 0.;
	    for (int i=0; i < n_users; i++) {
	    	mean_value += sample[i].value;
	    }
	    mean_value /= n_users;
	    
	    for (int userIndex = 0; userIndex < n_users; userIndex++) {
			userVectors[INDEX(userIndex, 0)] = mean_value;
			userVectors[INDEX(userIndex, USER_BIAS_INDEX)] = 0; // will store user bias
			userVectors[INDEX(userIndex, ITEM_BIAS_INDEX)] = 1; // corresponding item feature contains item bias
			for (int feature = FEATURE_OFFSET; feature < rank; feature++) {
				userVectors[INDEX(userIndex, feature)] = dist(engine) * NOISE;
			}
	    }
	    
	    for (int itemIndex = 0; itemIndex < n_items; itemIndex++) {
	      itemVectors[INDEX(itemIndex, 0)] = 1; // corresponding user feature contains global average
	      itemVectors[INDEX(itemIndex, USER_BIAS_INDEX)] = 1; // corresponding user feature contains user bias
	      itemVectors[INDEX(itemIndex, ITEM_BIAS_INDEX)] = 0; // will store item bias
	      for (int feature = FEATURE_OFFSET; feature < rank; feature++) {
	        itemVectors[INDEX(itemIndex, feature)] = dist(engine) * NOISE;
	      }
	    }
	  }
	
	void update(double preference, double *userVector, double *itemVector, double mu) {
		double prediction = dot(userVector, itemVector, rank);
		double err = preference - prediction;
				
		for(int i=FEATURE_OFFSET; i < rank; i++) {
			userVector[i] += mu * (err * itemVector[i] - lambda * userVector[i]);
		    itemVector[i] += mu * (err * userVector[i] - lambda * itemVector[i]);
		}
		
		userVector[USER_BIAS_INDEX] += biasMuRatio * mu * (err - biasLambdaRatio * lambda * userVector[USER_BIAS_INDEX]);
		itemVector[ITEM_BIAS_INDEX] += biasMuRatio * mu * (err - biasLambdaRatio * lambda * itemVector[ITEM_BIAS_INDEX]);
	}
	
	static double dot(double* userVector, double* itemVector, int rank) {
		double sum = 0;
		for (int k = 0; k < rank; k++) {
		  sum += userVector[k] * itemVector[k];
		}
		return sum;
	}
	
	
	public:
		double getValue(int user_id, int item_id) {
			return dot(&userVectors[INDEX(user_id, 0)], &itemVectors[INDEX(item_id, 0)], rank);
		}
	
		SgdMf(int n_iter, double lambda, int num_feature) {
			this->n_iter = n_iter;
			this->lambda = lambda;
			this->rank = num_feature + FEATURE_OFFSET;
		}
		
		~SgdMf() {
			delete[] userVectors;
			delete[] itemVectors;
		}
		
		void fit (LogData *sample, int n_log, int n_samples, int n_dimentions) {
			userVectors = new double[n_samples * rank];
			itemVectors = new double[n_dimentions * rank];
			initialize(sample, userVectors, itemVectors, n_samples, n_dimentions);
			for(int i=0; i < n_iter; i++) {
				double mu = mu0 * pow(decayFactor, i - 1) * pow(i + stepOffset, forgettingExponent);
				for(int j=0; j < n_log; j++) {
					int sample_idx = rand() % n_log;
					update(sample[sample_idx].value, &userVectors[INDEX(sample[sample_idx].user_id, 0)],
							&itemVectors[INDEX(sample[sample_idx].item_id, 0)], mu);
				}
				if (i % 10 == 0) {
					printf("%d / %d\n", i , n_iter);
				}
			}
		}
};

class MatrixFactrization2 {
	void fit(LogData *sample, int n_log, int n_user, int n_item) {
		int **word_indexes = new int*[n_user];
		unsigned short **word_counts = new unsigned short*[n_user];
		int *n_word_type_each_doc = new int[n_user];
		int *n_word_each_doc = new int[n_user];
		int n_all_word = 0;
		
		memset(n_word_type_each_doc, 0 , sizeof(int) * n_user);
		for (i=0; i < n_log; i++) {
			n_word_type_each_doc[sample[i].user_id]++;
		}
		
		memset(n_word_each_doc, 0 , sizeof(int) * n_user);
		int *cur_word = new int[n_user];
		memset(cur_word, 0 , sizeof(int) * n_user);
		for (i=0; i < n_log; i++) {
			n_word_each_doc[sample[i].user_id] += sample[i].n_purchase;
			n_all_word += sample[i].n_purchase;
			word_indexes[sample[i].user_id][cur_word[sample[i].user_id]] = sample[i].item_id;
			word_counts[sample[i].user_id][cur_word[sample[i].user_id]] = sample[i].user_id;
			cur_word[sample[i].user_id]++;
		}
		fclose(fp_word_indexes);
		
		Scvb0* sctx = scvb0Init(n_topics, 11000, 1600, 0.01, 0.001);
		scvb0Fit(sctx, word_indexes, word_counts, n_word_each_doc, n_word_type_each_doc, n_all_word, n_user, n_item);
		sctx->Theta;
	}
};

int main() {
	char buf[1024];
	int n_log = 0;
	int n_user = 0;
	int n_item = 0;
	int rank = 200;
	
	LogData *sample = new LogData[25000000];
	map<int, bool> unique_user, unique_item;
	FILE *fp = fopen("../data/" PROJECT "/train_data.csv", "r");
	while(fgets(buf, sizeof(buf), fp)){
		buf[strlen(buf) - 1] = '\0';
		char *ptr = strchr(buf, ',');
		*ptr = '\0';
		char *user_id = buf;
		char *item_id = ptr + 1;
		ptr = strchr(ptr + 1, ',');
		*ptr = '\0';
		char *n_purchase = ptr + 1;
		
		sample[n_log].user_id = atoi(user_id);
		sample[n_log].item_id = atoi(item_id);
		sample[n_log].value = atof(n_purchase);
		
		if (unique_user.find(sample[n_log].user_id) == unique_user.end()) {
			unique_user[sample[n_log].user_id] = true;
			if (sample[n_log].user_id > n_user) n_user = sample[n_log].user_id;
		}
		
		if (unique_item.find(sample[n_log].item_id) == unique_item.end()) {
			unique_item[sample[n_log].item_id] = true;
			if (sample[n_log].item_id > n_item) n_item = sample[n_log].item_id;
		}
		n_log++;
		
	}
	fclose(fp);
	n_user++;
	n_item++;
	
	printf("n_user:%d n_item:%d\n", n_user, n_item);
	puts("start");
	
	MatrixFactrization matrix_factriaztion(200, 1e-6, rank);
	matrix_factriaztion.fit(sample, n_log, n_user, n_item);
	
	fp = fopen("../data/" PROJECT "/train_data.csv", "r");
	double mse = 0.0;
	n_log = 0;
	while(fgets(buf, sizeof(buf), fp) ){
		if(n_log > 30000) break;
		buf[strlen(buf) - 1] = '\0';
		char *ptr = strchr(buf, ',');
		*ptr = '\0';
		char *user_id = buf;
		char *item_id = ptr + 1;
		ptr = strchr(ptr + 1, ',');
		*ptr = '\0';
		char *n_purchase = ptr + 1;
		if (atoi(user_id) >= n_user || atoi(item_id) >= n_item) continue;
		double est_value = matrix_factriaztion.getValue(atoi(user_id), atoi(item_id));
		printf("%f %f\n", atof(n_purchase), est_value);
		mse += (atof(n_purchase) - est_value) * (atof(n_purchase) - est_value);
		n_log++;
	}
	fclose(fp);
	printf("mse:%f\n", mse / n_log);
	delete[] sample;
	
}
