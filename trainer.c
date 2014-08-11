#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define DATA_SIZE 15000
#define TRAIN_DATA_SIZE data_data_size * 0.8
#define TEST_DATA_SIZE data_data_size * 0.2
#define ATTRI 16
#define NO_OF_INPUT 17
#define NO_OF_OUTPUT 26
#define INIT_WEIGHT ((((double)(rand()%1000))/1000)-0.5)
#define INPUT0  1

#define OUTPUT_LAYER_NODE 26

typedef struct NeuronLink{
	double weight;	//weight of this linking
	struct Neuron* in;	//in-comming neuron address(the neuron gives output to next layer)
	struct Neuron* out;	//out-going neuron address(the neuron gets input from previous layer)
} neuron_link;

typedef struct Neuron{
	int noi;	//no of in-comming neuron link
	int noo;	//no of out-going neuron link
	struct NeuronLink** in_neuron;	//an array of in-comming neuron link
	struct NeuronLink** out_neuron;	//an array of out-goning neuron link
	double weighted_sum; //summation of the products of weight and input value
	double output;	//output value
	double err;
	//error term: (T - O)g'(weighted_sum) in output layer,
	//			  g'(weighted_sum) times the summation of the products of
	//			  out-going weight and error term of that neuron for hidden/input layer
} neuron;

FILE * fptr;
int data_data_size = DATA_SIZE;
int class_data_size = DATA_SIZE;
int no_of_attr = ATTRI;
int no_of_output = NO_OF_OUTPUT;
int i, j, k, l;

//link up two neuron and return the link
neuron_link* link_up_neuron(neuron* in, neuron* out){
	neuron_link* link = (neuron_link*) malloc(sizeof(neuron_link));
	link->in = in;
	link->out = out;
	link->weight = INIT_WEIGHT;
	return link;
}

//update the output of a neuron
void neuron_update(neuron* node){
	int m;
	
	node->weighted_sum = 0;
	for(m = 0;m < node->noi;m++){
		node->weighted_sum += node->in_neuron[m]->weight * node->in_neuron[m]->in->output;
	}
	node->output = 1.0 / (1.0 + exp(-1.0 * node->weighted_sum));
	return;
}

//update the error of a neuron
void neuron_update_err(neuron* node){
	int m;
	double weighted_err = 0;
	double e_tmp = 0;
	
	for(m = 0;m < node->noo;m++){
		weighted_err += node->out_neuron[m]->weight * node->out_neuron[m]->out->err;
	}
	e_tmp = exp(node->weighted_sum);
	node->err = weighted_err * e_tmp / pow((e_tmp + 1), 2);
	return;
}

//update the weight of a neuron
void neuron_update_weight(neuron* node, double alpha){
	int m;
	
	for(m = 0;m < node->noi;m++){
		node->in_neuron[m]->weight += alpha * node->in_neuron[m]->in->output * node->err;
	}
	return;
}

double neural_networks(int** data, char* class, int no_of_h_layer, int no_of_h_node, double alpha, int no_of_iter){
	int no_of_hidden_layer = 1;//no of hidden layers
	int ans;
	int train_data_size = TRAIN_DATA_SIZE;
	int test_data_size = TEST_DATA_SIZE;
	int correct_flag = 0;
	int correct_cnt = 0;
	double accuracy;
	double weighted_err;
	double e_tmp;
	neuron input_data[NO_OF_INPUT];	//an array of neuron containing input value only, input[0] equals INPUT0, 16 attributes, thus no of input is 17
	neuron** hidden_neuron;	//hidden layer
	neuron output_neuron[OUTPUT_LAYER_NODE];	//output layer
	
	//printf("layer:[%d]node:[%d]a:[%f]i:[%d]train:[%d]test:[%d]\n", no_of_h_layer, no_of_h_node, alpha, no_of_iter, train_data_size, test_data_size);
	
	if(no_of_h_layer <= 0 || no_of_h_node <= 0){
		return 0;
	}
	
	//init hidden node
	hidden_neuron = (neuron**) malloc(sizeof(neuron*) * no_of_h_layer);
	for(i = 0;i < no_of_h_layer;i++){
		hidden_neuron[i] = (neuron*) malloc(sizeof(neuron) * no_of_h_node);
	}
	//init first layer of hidden layer
	for(j = 0;j < no_of_h_node;j++){
		hidden_neuron[0][j].noi = NO_OF_INPUT;
		hidden_neuron[0][j].noo = 0;
		hidden_neuron[0][j].in_neuron = (neuron_link**) malloc(sizeof(neuron_link*) * hidden_neuron[0][j].noi);
		if(no_of_h_layer == 1){
			hidden_neuron[0][j].out_neuron = (neuron_link**) malloc(sizeof(neuron_link*) * NO_OF_OUTPUT);
		}else{
			hidden_neuron[0][j].out_neuron = (neuron_link**) malloc(sizeof(neuron_link*) * no_of_h_node);
		}
		hidden_neuron[0][j].in_neuron[0] = link_up_neuron(&input_data[0], &hidden_neuron[0][j]);
		for(k = 1;k < hidden_neuron[0][j].noi;k++){
			hidden_neuron[0][j].in_neuron[k] = link_up_neuron(&input_data[k], &hidden_neuron[0][j]);
		}
	}
	//init the 2nd to no_of_h_layer layer of hidden layer
	for(i = 1;i < no_of_h_layer;i++){
		for(j = 0;j < no_of_h_node;j++){
			hidden_neuron[i][j].noi = no_of_h_node + 1;
			hidden_neuron[i][j].noo = 0;
			hidden_neuron[i][j].in_neuron = (neuron_link**) malloc(sizeof(neuron_link*) * hidden_neuron[i][j].noi);
			if(i == (no_of_h_layer - 1)){
				hidden_neuron[i][j].out_neuron = (neuron_link**) malloc(sizeof(neuron_link*) * NO_OF_OUTPUT);
			}else{
				hidden_neuron[i][j].out_neuron = (neuron_link**) malloc(sizeof(neuron_link*) * no_of_h_node);
			}
			hidden_neuron[i][j].in_neuron[0] = link_up_neuron(&input_data[0], &hidden_neuron[i][j]);
			for(k = 1;k < hidden_neuron[i][j].noi;k++){
				hidden_neuron[i][j].in_neuron[k] = link_up_neuron(&hidden_neuron[i - 1][k - 1], &hidden_neuron[i][j]);
				hidden_neuron[i - 1][k - 1].out_neuron[hidden_neuron[i - 1][k - 1].noo] = hidden_neuron[i][j].in_neuron[k];
				hidden_neuron[i - 1][k - 1].noo++;
			}
		}
	}
	//init output layer
	for(i = 0;i < OUTPUT_LAYER_NODE;i++){
		output_neuron[i].noi = no_of_h_node + 1;
		output_neuron[i].noo = 0;
		output_neuron[i].in_neuron = (neuron_link**) malloc(sizeof(neuron_link*) * output_neuron[i].noi);
		output_neuron[i].out_neuron = (neuron_link**) malloc(sizeof(neuron_link*) * NO_OF_OUTPUT);
		output_neuron[i].in_neuron[0] = link_up_neuron(&input_data[0], &output_neuron[i]);
		for(j = 1;j < output_neuron[i].noi;j++){
			output_neuron[i].in_neuron[j] = link_up_neuron(&hidden_neuron[no_of_h_layer - 1][j - 1], &output_neuron[i]);
			hidden_neuron[no_of_h_layer - 1][j - 1].out_neuron[hidden_neuron[no_of_h_layer - 1][j - 1].noo] = output_neuron[i].in_neuron[j];
			hidden_neuron[no_of_h_layer - 1][j - 1].noo++;
		}
	}
	
	//do training
	for(l = 0;l < no_of_iter;l++){
		for(i = 0;i < data_data_size;i++){
			//input data
			input_data[0].output = INPUT0;
			for(j = 1;j < NO_OF_INPUT;j++){
				input_data[j].output = (double) data[i][j - 1];
			}
			//update hidden layer			
			for(j = 0;j < no_of_h_layer;j++){
				for(k = 0;k < no_of_h_node;k++){
					neuron_update(&hidden_neuron[j][k]);
				}
			}
			//update output layer
			for(j = 0;j < OUTPUT_LAYER_NODE;j++){
				neuron_update(&output_neuron[j]);
			}
			//calculate the error of output layer
			ans = class[i] - 65;
			for(j = 0;j < OUTPUT_LAYER_NODE;j++){
				if(j == ans){
					output_neuron[j].err = 1.0 - output_neuron[j].output;
				}else{
					output_neuron[j].err = 0.0 - output_neuron[j].output;
				}
				e_tmp = exp(output_neuron[j].weighted_sum);
				output_neuron[j].err *= (e_tmp / pow((e_tmp + 1), 2));
			}
			//update the weight of output layer
			for(j = 0;j < OUTPUT_LAYER_NODE;j++){
				neuron_update_weight(&output_neuron[j], alpha);
			}
			//calculate the error and update the weight of hidden layer weighted_err
			for(j = (no_of_h_layer - 1);j >= 0;j--){
				for(k = 0;k < no_of_h_node;k++){
					neuron_update_err(&hidden_neuron[j][k]);
					neuron_update_weight(&hidden_neuron[j][k], alpha);
				}
			}
		}
		printf("iteration:[%d]\n", l);
	}
	//do testing
	for(i = train_data_size;i < data_data_size;i++){
		input_data[0].output = INPUT0;
		for(j = 1;j < NO_OF_INPUT;j++){
			input_data[j].output = (double) data[i][j - 1];
		}
		for(j = 0;j < no_of_h_layer;j++){
			for(k = 0;k < no_of_h_node;k++){
				neuron_update(&hidden_neuron[j][k]);
			}
		}
		for(j = 0;j < OUTPUT_LAYER_NODE;j++){
			neuron_update(&output_neuron[j]);
		}
		ans = class[i] - 65;
		//calculate the error of output layer
		correct_flag = 1;
		for(j = 0;j < OUTPUT_LAYER_NODE;j++){
			if(j == ans){
				if(output_neuron[j].output <= 0.5){
					correct_flag = 0;
					break;
				}
			}else{
				if(output_neuron[j].output > 0.5){
					correct_flag = 0;
					break;
				}
			}
		}
		if(correct_flag == 1){
			correct_cnt++;
		}
	}
	accuracy = (double)correct_cnt / test_data_size;

	//print out the result data
	fprintf(fptr, "%.1f\n", accuracy);
	fprintf(fptr, "%d\n", no_of_h_layer);
	//first hidden layer
	fprintf(fptr, "I %d H %d\n", no_of_attr, no_of_h_node);
	for(i = 0;i < no_of_h_node;i++){
		for(j = 0;j < hidden_neuron[0][i].noi;j++){
			fprintf(fptr, "%f ", hidden_neuron[0][i].in_neuron[j]->weight);
		}
		fprintf(fptr, "\n");
	}
	//2nd to no_of_h_layer hidden layer
	for(i = 1;i < no_of_h_layer;i++){
		fprintf(fptr, "H %d H %d\n", no_of_h_node, no_of_h_node);
		for(j = 0;j < no_of_h_node;j++){
			for(k = 0;k < hidden_neuron[i][j].noi;k++){
				fprintf(fptr, "%f ", hidden_neuron[i][j].in_neuron[k]->weight);
			}
			fprintf(fptr, "\n");
		}
	}
	//output layer
	fprintf(fptr, "H %d O %d\n", no_of_h_node, OUTPUT_LAYER_NODE);
	for(i = 0;i < OUTPUT_LAYER_NODE;i++){
		for(j = 0;j <= no_of_h_node;j++){
			fprintf(fptr, "%f ", output_neuron[i].in_neuron[j]->weight);
		}
		fprintf(fptr, "\n");
	}

	return accuracy;
}

int main(int argc, char** argv){
	int** data;
	char* class;
	char buff[10];
	double* nn_result;
	int hlayer, hnode, no_of_iter;
	double a, accuracy;

	if(argc < 4 || argc > 6){
		printf("Incorrect arguments.\n");
		exit(0);
	}
	
	//read data from dataset
	fptr = fopen(argv[1], "r");
	fscanf(fptr, "%d", &data_data_size);
	fscanf(fptr, "%d", &no_of_attr);
	data = (int**) malloc(sizeof(int*) * data_data_size);
	for(i = 0;i < data_data_size;i++){
		data[i] = (int*) malloc(sizeof(int) * no_of_attr);
	}
	for(i = 0;i < data_data_size;i++){
		for(j = 0;j < no_of_attr;j++){
			fscanf(fptr, "%d", &data[i][j]);
		}
	}
	fclose(fptr);
	//read answers for each record
	fptr = fopen(argv[2], "r");
	//fscanf(fptr, "%d\n", &class_data_size);
		fgets(buff, 10, fptr);
		class_data_size = atoi(buff);
	//check whether no of answer matches no of records in dataset
	if(class_data_size != data_data_size){
		printf("'class.dat' data size does not match 'data.data' data size\n");
		exit(0);
	}
	//read the answer
	class = (char*) malloc(sizeof(char*) * class_data_size);
	for(i = 0;i < class_data_size;i++){
		fgets(buff, 10, fptr);
		class[i] = buff[0];
		//fscanf(fptr, "%c", &class[i]);
	}
	fclose(fptr);
	no_of_iter = 1000;
	if(argc > 4){
		if(no_of_iter > atoi(argv[4])){
			no_of_iter = atoi(argv[4]);
		}
	}
	if(argc > 5){
		if(atoi(argv[5]) == 1){
			no_of_iter = 300;
		}
	}
	hlayer = 2;
	hnode = 40;
	a = 0.0875;
	// for(hlayer = 2;hlayer <= 5;hlayer++){
		// for(hnode = 25;hnode <= 151;hnode += 25){
			// for(a = 0.05;a < 0.0876;a += 0.0125){
				// fptr = fopen(argv[3], "w");
				// accuracy = neural_networks(data, class, hlayer, hnode, a, no_of_iter);
				// printf("layer:[%d]node:[%d]alpha:[%f]no_of_iter:[%d]accuracy:[%f]\n", hlayer, hnode, a, no_of_iter, accuracy);
				// fclose(fptr);
			// }
		// }
	// }
	fptr = fopen(argv[3], "w");
	if(fptr == NULL)
		printf("file cannot open.\n");
	accuracy = neural_networks(data, class, hlayer, hnode, a, no_of_iter);
	printf("layer:[%d]node:[%d]alpha:[%f]no_of_iter:[%d]accuracy:[%f]\n", hlayer, hnode, a, no_of_iter, accuracy);
	fclose(fptr);
	return;
}