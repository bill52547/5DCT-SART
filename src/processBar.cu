#include "processBar.cuh"
void processBar(int i1, int n1, int i2, int n2)
{
    mexEvalString("clc;");  
    int N = n1 * n2, step = 1, id = i1 + i2 * n1 + 1;
    while (N / step > 40)
        step++;
    float perc = float(id) * 100 / (N);
    mexPrintf("%3.1f %% [", perc);
    for (int i = 0; i < N; i += step)
    {
        if (i <= id)
        {mexPrintf("=");}            
        else
        {mexPrintf("-");}            
    }
    mexPrintf("] step = %d\n", step);

    mexPrintf("Iter (%d / %d), Frac (%d / %d), Total (%d / %d).\n", i1 + 1, n1, i2 + 1, n2, id, N); mexEvalString("drawnow;");
}