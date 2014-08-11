#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define DATA_SIZE 15000
#define ATTRI_NUM 16

int attri[DATA_SIZE][ATTRI_NUM];
char letter[DATA_SIZE];
char tmp[DATA_SIZE][400];

int main(int argc, char** argv){
	FILE * fptr;
	FILE * fptr1;
	char *attri_tmp;
	int i = 0, j = 0, r =0;
	
	if(argc != 4){
		printf("Incorrect arguments.\n");
		exit(0);
	}
	
	fptr = fopen(argv[1], "r");
	while(fgets(tmp[i], 1000, fptr) != NULL){
		i++;
	}
	fclose(fptr);
	
	fptr = fopen(argv[2], "w");
	fprintf(fptr, "%d %d\n", i, ATTRI_NUM);
	for(j = 0;j < i;j++){
		for(r = 2;r < strlen(tmp[j]);r++){
			if(tmp[j][r] == ','){
				fprintf(fptr, " ");
			}else
				fprintf(fptr, "%c", tmp[j][r]);
		}
	}
	fclose(fptr);
	
	fptr = fopen(argv[3], "w");
	fprintf(fptr, "%d\n", i);
	for(j = 0;j < i;j++){
		fprintf(fptr, "%c\n", tmp[j][0]);
	}
	fclose(fptr);
	
}