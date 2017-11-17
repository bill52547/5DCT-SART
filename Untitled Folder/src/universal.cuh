#ifndef _UNIVERSAL_H
#define _UNIVERSAL_H
#include <math.h>
#include "mex.h"

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MAX6(a, b, c, d, e, f) MAX(MAX(MAX(a, b), MAX(c, d)), MAX(e, f))

//#define MAX4(a, b, c, d) (((((a) > (b)) ? (a) : (b)) > (((c) > (d)) ? (c) : (d))) > (((a) > (b)) ? (a) : (b)) : (((c) > (d)) ? (c) : (d)))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
#define MIN6(a, b, c, d, e, f) MIN(MIN(MIN(a, b), MIN(c, d)), MIN(e, f))

#define ABS(x) ((x) > 0 ? (x) : -(x))
#define PI 3.141592653589793f

#define BLOCKWIDTH 16
#define BLOCKHEIGHT 16 
#define BLOCKDEPTH 4

#endif // _UNIVERSAL_H