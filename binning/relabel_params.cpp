#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <cmath>


#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <cmath>


using namespace std;
vector<vector<double> > allDatamult;


int main(int argc, char* argv[]) {

  ifstream fin("lcdm.txt");
  string line;
  while (getline(fin, line)) {      // for each line
          vector<double> lineData;           // create a new row
          double val;
          istringstream lineStream(line);
          while (lineStream >> val) {          // for each value in line
                  lineData.push_back(val);           // add to the current row
          }
          allDatamult.push_back(lineData);         // add row to allData
  }


//output file name
const char* output = "lcdm_mnu_ordered.txt";

    /* Open output file */
FILE* fp = fopen(output, "w");

int Ncos= allDatamult.size();
printf("%d \n ", Ncos);
int temp;
double neworder[11];

// Index of parameter we want to order
int paramindex = 3; 

for(int i =0; i < Ncos;  i ++) {

  for(int j=i+1; j<Ncos; j++){
      if (allDatamult[j][paramindex]<allDatamult[i][paramindex]) {

        for(int k=0; k< 11; k++){
         neworder[k] = allDatamult[i][k];
         allDatamult[i][k] = allDatamult[j][k];
         allDatamult[j][k] = neworder[k] ;
           }
         }
       }
     }

for(int j =0; j < Ncos;  j ++) {
  printf("%d ", int(allDatamult[j][0]) );
 }


for(int j =0; j < Ncos;  j ++) {

  fprintf(fp,"%d ", j+1);

    for(int i=1; i< 11; i++){
        fprintf(fp,"%e ", allDatamult[j][i]); // print to file
       }

       fprintf(fp,"\n");
}


	/*close output file*/
    fclose(fp);
    return 0;
}
