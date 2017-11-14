#include "processBar.h"
void processBar(int i1, int n1, int i2, int n2)
{
    // clrscr();
    mexEvalString("clc;");mexEvalString("drawnow;");
    mexPrintf("Total %d Iterations: [", n1);mexEvalString("drawnow;");
    for (int i = 0; i < n1; i++)
    {
        if (i <= i1)
        {mexPrintf("=");mexEvalString("drawnow;");}            
        else
        {mexPrintf(".");mexEvalString("drawnow;");}            
    }
    mexPrintf("]%d\n", i1);mexEvalString("drawnow;");
    mexPrintf("Total %d Fractions:  [", n2);mexEvalString("drawnow;");
    for (int i = 0; i < n2; i++)
    {
        if (i <= i2)
        {mexPrintf("=");mexEvalString("drawnow;");}            
        else
        {mexPrintf(".");mexEvalString("drawnow;");}            
    }
    mexPrintf("]%d\n", i2);mexEvalString("drawnow;");
}