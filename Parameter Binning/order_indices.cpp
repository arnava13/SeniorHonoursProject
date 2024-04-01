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
    ifstream fin("YOUR INPUT COSMOLOGY FILE FOR ENTIRE DATASET.txt");
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

    // New output file for ordered indices
    ofstream fout("YOUR OUTPUT FILE WITH THE DESIRED ORDER OF ORIGINAL INDICES.txt");

    int Ncos = allDatamult.size();
    cout << Ncos << "\n"; // Originally printed to console
    int temp;
    double neworder[11];

    // Index of parameter we want to order
    int paramindex = "INDEX OF THE PARAMETER TO ORDER UPON";

    for(int i = 0; i < Ncos; i++) {
        for(int j = i + 1; j < Ncos; j++) {
            if (allDatamult[j][paramindex] < allDatamult[i][paramindex]) {
                for(int k = 0; k < 11; k++) {
                    neworder[k] = allDatamult[i][k];
                    allDatamult[i][k] = allDatamult[j][k];
                    allDatamult[j][k] = neworder[k];
                }
            }
        }
    }

    // Write ordered indices to file
    for(int j = 0; j < Ncos; j++) {
        fout << int(allDatamult[j][0]) << " ";
    }

    // Close output file
    fout.close();
    return 0;
}
