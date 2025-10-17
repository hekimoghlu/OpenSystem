/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#ifndef __SUPERLU_SCOMPLEX /* allow multiple inclusions */
#define __SUPERLU_SCOMPLEX


#ifndef SCOMPLEX_INCLUDE
#define SCOMPLEX_INCLUDE

typedef struct { float r, i; } complex;


/* Macro definitions */

/*! \brief Complex Addition c = a + b */
#define c_add(c, a, b) { (c)->r = (a)->r + (b)->r; \
			 (c)->i = (a)->i + (b)->i; }

/*! \brief Complex Subtraction c = a - b */
#define c_sub(c, a, b) { (c)->r = (a)->r - (b)->r; \
			 (c)->i = (a)->i - (b)->i; }

/*! \brief Complex-Double Multiplication */
#define cs_mult(c, a, b) { (c)->r = (a)->r * (b); \
                           (c)->i = (a)->i * (b); }

/*! \brief Complex-Complex Multiplication */
#define cc_mult(c, a, b) { \
	float cr, ci; \
    	cr = (a)->r * (b)->r - (a)->i * (b)->i; \
    	ci = (a)->i * (b)->r + (a)->r * (b)->i; \
    	(c)->r = cr; \
    	(c)->i = ci; \
    }

#define cc_conj(a, b) { \
        (a)->r = (b)->r; \
        (a)->i = -((b)->i); \
    }

/*! \brief Complex equality testing */
#define c_eq(a, b)  ( (a)->r == (b)->r && (a)->i == (b)->i )


#ifdef __cplusplus
extern "C" {
#endif

/* Prototypes for functions in scomplex.c */
void c_div(complex *, complex *, complex *);
double c_abs(complex *);     /* exact */
double c_abs1(complex *);    /* approximate */
void c_exp(complex *, complex *);
void r_cnjg(complex *, complex *);
double r_imag(complex *);
complex c_sgn(complex *);
complex c_sqrt(complex *);



#ifdef __cplusplus
  }
#endif

#endif

#endif  /* __SUPERLU_SCOMPLEX */
