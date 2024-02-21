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

/* Example code to output the 1-loop powerspectrum for modified gravity in real and redshift space*/

int main(int argc, char* argv[]) {

  ifstream fin("lcdm_mnu_ordered.txt");
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
const char* outputA = "binA.txt";
const char* outputB = "binB.txt";
const char* outputC = "binC.txt";
const char* outputD = "binD.txt";

    /* Open output file */
FILE* fpA = fopen(outputA, "w");
FILE* fpB = fopen(outputB, "w");
FILE* fpC = fopen(outputC, "w");
FILE* fpD = fopen(outputD, "w");

int Ncos= (allDatamult.size());
int binsize = Ncos/4;
printf("%d", Ncos);

for(int j =0; j < binsize;  j ++) {
  fprintf(fpA,"%d ", j+1);
    for(int i=1; i< 11; i++){
        fprintf(fpA,"%e ", allDatamult[j][i]); // print to file
       }
       fprintf(fpA,"\n");
}
fclose(fpA);

for(int j =binsize; j < 2*binsize;  j ++) {
  fprintf(fpB,"%d ", j+1-binsize);
    for(int i=1; i< 11; i++){
        fprintf(fpB,"%e ", allDatamult[j][i]); // print to file
       }
       fprintf(fpB,"\n");
}
fclose(fpB);


for(int j = 2*binsize; j < 3*binsize;  j ++) {
  fprintf(fpC,"%d ", j+1-2*binsize);
    for(int i=1; i< 11; i++){
        fprintf(fpC,"%e ", allDatamult[j][i]); // print to file
       }
       fprintf(fpC,"\n");
}
fclose(fpC);


for(int j =3*binsize; j < 4*binsize;  j ++) {
  fprintf(fpD,"%d ", j+1-3*binsize);
    for(int i=1; i< 11; i++){
        fprintf(fpD,"%e ", allDatamult[j][i]); // print to file
       }
       fprintf(fpD,"\n");
}
fclose(fpD);


    return 0;
}
