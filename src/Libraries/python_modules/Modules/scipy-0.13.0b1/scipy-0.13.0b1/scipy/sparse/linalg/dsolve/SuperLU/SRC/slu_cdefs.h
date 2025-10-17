/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#ifndef __SUPERLU_cSP_DEFS /* allow multiple inclusions */
#define __SUPERLU_cSP_DEFS

/*
 * File name:		csp_defs.h
 * Purpose:             Sparse matrix types and function prototypes
 * History:
 */

#ifdef _CRAY
#include <fortran.h>
#include <string.h>
#endif

/* Define my integer type int_t */
typedef int int_t; /* default */

#include <math.h>
#include <limits.h>
#include "slu_Cnames.h"
#include "supermatrix.h"
#include "slu_util.h"
#include "slu_scomplex.h"



typedef struct {
    int     *xsup;    /* supernode and column mapping */
    int     *supno;   
    int     *lsub;    /* compressed L subscripts */
    int	    *xlsub;
    complex  *lusup;   /* L supernodes */
    int     *xlusup;
    complex  *ucol;    /* U columns */
    int     *usub;
    int	    *xusub;
    int     nzlmax;   /* current max size of lsub */
    int     nzumax;   /*    "    "    "      ucol */
    int     nzlumax;  /*    "    "    "     lusup */
    int     n;        /* number of columns in the matrix */
    LU_space_t MemModel; /* 0 - system malloc'd; 1 - user provided */
    int     num_expansions;
    ExpHeader *expanders; /* Array of pointers to 4 types of memory */
    LU_stack_t stack;     /* use user supplied memory */
} GlobalLU_t;


/* -------- Prototypes -------- */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Driver routines */
extern void
cgssv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
cgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, float *, float *, SuperMatrix *, SuperMatrix *,
       void *, int, SuperMatrix *, SuperMatrix *,
       float *, float *, float *, float *,
       mem_usage_t *, SuperLUStat_t *, int *);
    /* ILU */
extern void
cgsisv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
cgsisx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, float *, float *, SuperMatrix *, SuperMatrix *,
       void *, int, SuperMatrix *, SuperMatrix *, float *, float *,
       mem_usage_t *, SuperLUStat_t *, int *);


/*! \brief Supernodal LU factor related */
extern void
cCreate_CompCol_Matrix(SuperMatrix *, int, int, int, complex *,
		       int *, int *, Stype_t, Dtype_t, Mtype_t);
extern void
cCreate_CompRow_Matrix(SuperMatrix *, int, int, int, complex *,
		       int *, int *, Stype_t, Dtype_t, Mtype_t);
extern void
cCopy_CompCol_Matrix(SuperMatrix *, SuperMatrix *);
extern void
cCreate_Dense_Matrix(SuperMatrix *, int, int, complex *, int,
		     Stype_t, Dtype_t, Mtype_t);
extern void
cCreate_SuperNode_Matrix(SuperMatrix *, int, int, int, complex *, 
		         int *, int *, int *, int *, int *,
			 Stype_t, Dtype_t, Mtype_t);
extern void
cCopy_Dense_Matrix(int, int, complex *, int, complex *, int);

extern void    countnz (const int, int *, int *, int *, GlobalLU_t *);
extern void    ilu_countnz (const int, int *, int *, GlobalLU_t *);
extern void    fixupL (const int, const int *, GlobalLU_t *);

extern void    callocateA (int, int, complex **, int **, int **);
extern void    cgstrf (superlu_options_t*, SuperMatrix*,
                       int, int, int*, void *, int, int *, int *, 
                       SuperMatrix *, SuperMatrix *, SuperLUStat_t*, int *);
extern int     csnode_dfs (const int, const int, const int *, const int *,
			     const int *, int *, int *, GlobalLU_t *);
extern int     csnode_bmod (const int, const int, const int, complex *,
                              complex *, GlobalLU_t *, SuperLUStat_t*);
extern void    cpanel_dfs (const int, const int, const int, SuperMatrix *,
			   int *, int *, complex *, int *, int *, int *,
			   int *, int *, int *, int *, GlobalLU_t *);
extern void    cpanel_bmod (const int, const int, const int, const int,
                           complex *, complex *, int *, int *,
			   GlobalLU_t *, SuperLUStat_t*);
extern int     ccolumn_dfs (const int, const int, int *, int *, int *, int *,
			   int *, int *, int *, int *, int *, GlobalLU_t *);
extern int     ccolumn_bmod (const int, const int, complex *,
			   complex *, int *, int *, int,
                           GlobalLU_t *, SuperLUStat_t*);
extern int     ccopy_to_ucol (int, int, int *, int *, int *,
                              complex *, GlobalLU_t *);         
extern int     cpivotL (const int, const double, int *, int *, 
                         int *, int *, int *, GlobalLU_t *, SuperLUStat_t*);
extern void    cpruneL (const int, const int *, const int, const int,
			  const int *, const int *, int *, GlobalLU_t *);
extern void    creadmt (int *, int *, int *, complex **, int **, int **);
extern void    cGenXtrue (int, int, complex *, int);
extern void    cFillRHS (trans_t, int, complex *, int, SuperMatrix *,
			  SuperMatrix *);
extern void    cgstrs (trans_t, SuperMatrix *, SuperMatrix *, int *, int *,
                        SuperMatrix *, SuperLUStat_t*, int *);
/* ILU */
extern void    cgsitrf (superlu_options_t*, SuperMatrix*, int, int, int*,
		        void *, int, int *, int *, SuperMatrix *, SuperMatrix *,
                        SuperLUStat_t*, int *);
extern int     cldperm(int, int, int, int [], int [], complex [],
                        int [],	float [], float []);
extern int     ilu_csnode_dfs (const int, const int, const int *, const int *,
			       const int *, int *, GlobalLU_t *);
extern void    ilu_cpanel_dfs (const int, const int, const int, SuperMatrix *,
			       int *, int *, complex *, float *, int *, int *,
			       int *, int *, int *, int *, GlobalLU_t *);
extern int     ilu_ccolumn_dfs (const int, const int, int *, int *, int *,
				int *, int *, int *, int *, int *,
				GlobalLU_t *);
extern int     ilu_ccopy_to_ucol (int, int, int *, int *, int *,
                                  complex *, int, milu_t, double, int,
                                  complex *, int *, GlobalLU_t *, float *);
extern int     ilu_cpivotL (const int, const double, int *, int *, int, int *,
			    int *, int *, int *, double, milu_t,
                            complex, GlobalLU_t *, SuperLUStat_t*);
extern int     ilu_cdrop_row (superlu_options_t *, int, int, double,
                              int, int *, double *, GlobalLU_t *, 
                              float *, float *, int);


/*! \brief Driver related */

extern void    cgsequ (SuperMatrix *, float *, float *, float *,
			float *, float *, int *);
extern void    claqgs (SuperMatrix *, float *, float *, float,
                        float, float, char *);
extern void    cgscon (char *, SuperMatrix *, SuperMatrix *, 
		         float, float *, SuperLUStat_t*, int *);
extern float   cPivotGrowth(int, SuperMatrix *, int *, 
                            SuperMatrix *, SuperMatrix *);
extern void    cgsrfs (trans_t, SuperMatrix *, SuperMatrix *,
                       SuperMatrix *, int *, int *, char *, float *, 
                       float *, SuperMatrix *, SuperMatrix *,
                       float *, float *, SuperLUStat_t*, int *);

extern int     sp_ctrsv (char *, char *, char *, SuperMatrix *,
			SuperMatrix *, complex *, SuperLUStat_t*, int *);
extern int     sp_cgemv (char *, complex, SuperMatrix *, complex *,
			int, complex, complex *, int);

extern int     sp_cgemm (char *, char *, int, int, int, complex,
			SuperMatrix *, complex *, int, complex, 
			complex *, int);
extern         float slamch_(char *);


/*! \brief Memory-related */
extern int     cLUMemInit (fact_t, void *, int, int, int, int, int,
                            float, SuperMatrix *, SuperMatrix *,
                            GlobalLU_t *, int **, complex **);
extern void    cSetRWork (int, int, complex *, complex **, complex **);
extern void    cLUWorkFree (int *, complex *, GlobalLU_t *);
extern int     cLUMemXpand (int, int, MemType, int *, GlobalLU_t *);

extern complex  *complexMalloc(int);
extern complex  *complexCalloc(int);
extern float  *floatMalloc(int);
extern float  *floatCalloc(int);
extern int     cmemory_usage(const int, const int, const int, const int);
extern int     cQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
extern int     ilu_cQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);

/*! \brief Auxiliary routines */
extern void    creadhb(int *, int *, int *, complex **, int **, int **);
extern void    creadrb(int *, int *, int *, complex **, int **, int **);
extern void    creadtriple(int *, int *, int *, complex **, int **, int **);
extern void    cCompRow_to_CompCol(int, int, int, complex*, int*, int*,
		                   complex **, int **, int **);
extern void    cfill (complex *, int, complex);
extern void    cinf_norm_error (int, SuperMatrix *, complex *);
extern void    PrintPerf (SuperMatrix *, SuperMatrix *, mem_usage_t *,
			 complex, complex, complex *, complex *, char *);
extern float  sqselect(int, float *, int);


/*! \brief Routines for debugging */
extern void    cPrint_CompCol_Matrix(char *, SuperMatrix *);
extern void    cPrint_SuperNode_Matrix(char *, SuperMatrix *);
extern void    cPrint_Dense_Matrix(char *, SuperMatrix *);
extern void    cprint_lu_col(char *, int, int, int *, GlobalLU_t *);
extern int     print_double_vec(char *, int, double *);
extern void    check_tempv(int, complex *);

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_cSP_DEFS */

