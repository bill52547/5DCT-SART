#include "processBar.h"
void processBar(int i1, int n1, int i2, int n2)
{
    // clrscr();
    mexEvalString("clc;");mexEvalString("drawnow;");    
    mexPrintf("Iter (%d / %d), Frac (%d / %d):\n", i1, n1, i2, n2);mexEvalString("drawnow;");
    mexPrintf("(%d / %d) \n[", i1 * n2 + i2 + 1, n1 * n2);mexEvalString("drawnow;");
    for (int i = 0; i < n1 * n2; i++)
    {
        if (i <= i1 * n2 + i2)
        {mexPrintf("=");mexEvalString("drawnow;");}            
        else
        {mexPrintf("-");mexEvalString("drawnow;");}            
    }
    mexPrintf("]");mexEvalString("drawnow;");

}