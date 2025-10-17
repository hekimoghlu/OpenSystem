/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "slu_scomplex.h"


/*! \brief Complex Division c = a/b */
void c_div(complex *c, complex *a, complex *b)
{
    float ratio, den;
    float abr, abi, cr, ci;
  
    if( (abr = b->r) < 0.)
	abr = - abr;
    if( (abi = b->i) < 0.)
	abi = - abi;
    if( abr <= abi ) {
	if (abi == 0) {
	    fprintf(stderr, "z_div.c: division by zero\n");
            exit(-1);
	}	  
	ratio = b->r / b->i ;
	den = b->i * (1 + ratio*ratio);
	cr = (a->r*ratio + a->i) / den;
	ci = (a->i*ratio - a->r) / den;
    } else {
	ratio = b->i / b->r ;
	den = b->r * (1 + ratio*ratio);
	cr = (a->r + a->i*ratio) / den;
	ci = (a->i - a->r*ratio) / den;
    }
    c->r = cr;
    c->i = ci;
}


/*! \brief Returns sqrt(z.r^2 + z.i^2) */
double c_abs(complex *z)
{
    float temp;
    float real = z->r;
    float imag = z->i;

    if (real < 0) real = -real;
    if (imag < 0) imag = -imag;
    if (imag > real) {
	temp = real;
	real = imag;
	imag = temp;
    }
    if ((real+imag) == real) return(real);
  
    temp = imag/real;
    temp = real*sqrt(1.0 + temp*temp);  /*overflow!!*/
    return (temp);
}


/*! \brief Approximates the abs. Returns abs(z.r) + abs(z.i) */
double c_abs1(complex *z)
{
    float real = z->r;
    float imag = z->i;
  
    if (real < 0) real = -real;
    if (imag < 0) imag = -imag;

    return (real + imag);
}

/*! \brief Return the exponentiation */
void c_exp(complex *r, complex *z)
{
    float expx;

    expx = exp(z->r);
    r->r = expx * cos(z->i);
    r->i = expx * sin(z->i);
}

/*! \brief Return the complex conjugate */
void r_cnjg(complex *r, complex *z)
{
    r->r = z->r;
    r->i = -z->i;
}

/*! \brief Return the imaginary part */
double r_imag(complex *z)
{
    return (z->i);
}


/*! \brief SIGN functions for complex number. Returns z/abs(z) */
complex c_sgn(complex *z)
{
    register float t = c_abs(z);
    register complex retval;

    if (t == 0.0) {
	retval.r = 1.0, retval.i = 0.0;
    } else {
	retval.r = z->r / t, retval.i = z->i / t;
    }

    return retval;
}

/*! \brief Square-root of a complex number. */
complex c_sqrt(complex *z)
{
    complex retval;
    register float cr, ci, real, imag;

    real = z->r;
    imag = z->i;

    if ( imag == 0.0 ) {
        retval.r = sqrt(real);
        retval.i = 0.0;
    } else {
        ci = (sqrt(real*real + imag*imag) - real) / 2.0;
        ci = sqrt(ci);
        cr = imag / (2.0 * ci);
        retval.r = cr;
        retval.i = ci;
    }

    return retval;
}


