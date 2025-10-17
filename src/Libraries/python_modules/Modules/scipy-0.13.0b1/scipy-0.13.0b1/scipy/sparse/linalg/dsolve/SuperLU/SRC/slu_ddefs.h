/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#ifndef __SUPERLU_dSP_DEFS /* allow multiple inclusions */
#define __SUPERLU_dSP_DEFS

/*
 * File name:		dsp_defs.h
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



typedef struct {
    int     *xsup;    /* supernode and column mapping */
    int     *supno;   
    int     *lsub;    /* compressed L subscripts */
    int	    *xlsub;
    double  *lusup;   /* L supernodes */
    int     *xlusup;
    double  *ucol;    /* U columns */
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
dgssv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
dgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, double *, double *, SuperMatrix *, SuperMatrix *,
       void *, int, SuperMatrix *, SuperMatrix *,
       double *, double *, double *, double *,
       mem_usage_t *, SuperLUStat_t *, int *);
    /* ILU */
extern void
dgsisv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
dgsisx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, double *, double *, SuperMatrix *, SuperMatrix *,
       void *, int, SuperMatrix *, SuperMatrix *, double *, double *,
       mem_usage_t *, SuperLUStat_t *, int *);


/*! \brief Supernodal LU factor related */
extern void
dCreate_CompCol_Matrix(SuperMatrix *, int, int, int, double *,
		       int *, int *, Stype_t, Dtype_t, Mtype_t);
extern void
dCreate_CompRow_Matrix(SuperMatrix *, int, int, int, double *,
		       int *, int *, Stype_t, Dtype_t, Mtype_t);
extern void
dCopy_CompCol_Matrix(SuperMatrix *, SuperMatrix *);
extern void
dCreate_Dense_Matrix(SuperMatrix *, int, int, double *, int,
		     Stype_t, Dtype_t, Mtype_t);
extern void
dCreate_SuperNode_Matrix(SuperMatrix *, int, int, int, double *, 
		         int *, int *, int *, int *, int *,
			 Stype_t, Dtype_t, Mtype_t);
extern void
dCopy_Dense_Matrix(int, int, double *, int, double *, int);

extern void    countnz (const int, int *, int *, int *, GlobalLU_t *);
extern void    ilu_countnz (const int, int *, int *, GlobalLU_t *);
extern void    fixupL (const int, const int *, GlobalLU_t *);

extern void    dallocateA (int, int, double **, int **, int **);
extern void    dgstrf (superlu_options_t*, SuperMatrix*,
                       int, int, int*, void *, int, int *, int *, 
                       SuperMatrix *, SuperMatrix *, SuperLUStat_t*, int *);
extern int     dsnode_dfs (const int, const int, const int *, const int *,
			     const int *, int *, int *, GlobalLU_t *);
extern int     dsnode_bmod (const int, const int, const int, double *,
                              double *, GlobalLU_t *, SuperLUStat_t*);
extern void    dpanel_dfs (const int, const int, const int, SuperMatrix *,
			   int *, int *, double *, int *, int *, int *,
			   int *, int *, int *, int *, GlobalLU_t *);
extern void    dpanel_bmod (const int, const int, const int, const int,
                           double *, double *, int *, int *,
			   GlobalLU_t *, SuperLUStat_t*);
extern int     dcolumn_dfs (const int, const int, int *, int *, int *, int *,
			   int *, int *, int *, int *, int *, GlobalLU_t *);
extern int     dcolumn_bmod (const int, const int, double *,
			   double *, int *, int *, int,
                           GlobalLU_t *, SuperLUStat_t*);
extern int     dcopy_to_ucol (int, int, int *, int *, int *,
                              double *, GlobalLU_t *);         
extern int     dpivotL (const int, const double, int *, int *, 
                         int *, int *, int *, GlobalLU_t *, SuperLUStat_t*);
extern void    dpruneL (const int, const int *, const int, const int,
			  const int *, const int *, int *, GlobalLU_t *);
extern void    dreadmt (int *, int *, int *, double **, int **, int **);
extern void    dGenXtrue (int, int, double *, int);
extern void    dFillRHS (trans_t, int, double *, int, SuperMatrix *,
			  SuperMatrix *);
extern void    dgstrs (trans_t, SuperMatrix *, SuperMatrix *, int *, int *,
                        SuperMatrix *, SuperLUStat_t*, int *);
/* ILU */
extern void    dgsitrf (superlu_options_t*, SuperMatrix*, int, int, int*,
		        void *, int, int *, int *, SuperMatrix *, SuperMatrix *,
                        SuperLUStat_t*, int *);
extern int     dldperm(int, int, int, int [], int [], double [],
                        int [],	double [], double []);
extern int     ilu_dsnode_dfs (const int, const int, const int *, const int *,
			       const int *, int *, GlobalLU_t *);
extern void    ilu_dpanel_dfs (const int, const int, const int, SuperMatrix *,
			       int *, int *, double *, double *, int *, int *,
			       int *, int *, int *, int *, GlobalLU_t *);
extern int     ilu_dcolumn_dfs (const int, const int, int *, int *, int *,
				int *, int *, int *, int *, int *,
				GlobalLU_t *);
extern int     ilu_dcopy_to_ucol (int, int, int *, int *, int *,
                                  double *, int, milu_t, double, int,
                                  double *, int *, GlobalLU_t *, double *);
extern int     ilu_dpivotL (const int, const double, int *, int *, int, int *,
			    int *, int *, int *, double, milu_t,
                            double, GlobalLU_t *, SuperLUStat_t*);
extern int     ilu_ddrop_row (superlu_options_t *, int, int, double,
                              int, int *, double *, GlobalLU_t *, 
                              double *, double *, int);


/*! \brief Driver related */

extern void    dgsequ (SuperMatrix *, double *, double *, double *,
			double *, double *, int *);
extern void    dlaqgs (SuperMatrix *, double *, double *, double,
                        double, double, char *);
extern void    dgscon (char *, SuperMatrix *, SuperMatrix *, 
		         double, double *, SuperLUStat_t*, int *);
extern double   dPivotGrowth(int, SuperMatrix *, int *, 
                            SuperMatrix *, SuperMatrix *);
extern void    dgsrfs (trans_t, SuperMatrix *, SuperMatrix *,
                       SuperMatrix *, int *, int *, char *, double *, 
                       double *, SuperMatrix *, SuperMatrix *,
                       double *, double *, SuperLUStat_t*, int *);

extern int     sp_dtrsv (char *, char *, char *, SuperMatrix *,
			SuperMatrix *, double *, SuperLUStat_t*, int *);
extern int     sp_dgemv (char *, double, SuperMatrix *, double *,
			int, double, double *, int);

extern int     sp_dgemm (char *, char *, int, int, int, double,
			SuperMatrix *, double *, int, double, 
			double *, int);
extern         double dlamch_(char *);


/*! \brief Memory-related */
extern int     dLUMemInit (fact_t, void *, int, int, int, int, int,
                            double, SuperMatrix *, SuperMatrix *,
                            GlobalLU_t *, int **, double **);
extern void    dSetRWork (int, int, double *, double **, double **);
extern void    dLUWorkFree (int *, double *, GlobalLU_t *);
extern int     dLUMemXpand (int, int, MemType, int *, GlobalLU_t *);

extern double  *doubleMalloc(int);
extern double  *doubleCalloc(int);
extern int     dmemory_usage(const int, const int, const int, const int);
extern int     dQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
extern int     ilu_dQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);

/*! \brief Auxiliary routines */
extern void    dreadhb(int *, int *, int *, double **, int **, int **);
extern void    dreadrb(int *, int *, int *, double **, int **, int **);
extern void    dreadtriple(int *, int *, int *, double **, int **, int **);
extern void    dCompRow_to_CompCol(int, int, int, double*, int*, int*,
		                   double **, int **, int **);
extern void    dfill (double *, int, double);
extern void    dinf_norm_error (int, SuperMatrix *, double *);
extern void    PrintPerf (SuperMatrix *, SuperMatrix *, mem_usage_t *,
			 double, double, double *, double *, char *);
extern double  dqselect(int, double *, int);


/*! \brief Routines for debugging */
extern void    dPrint_CompCol_Matrix(char *, SuperMatrix *);
extern void    dPrint_SuperNode_Matrix(char *, SuperMatrix *);
extern void    dPrint_Dense_Matrix(char *, SuperMatrix *);
extern void    dprint_lu_col(char *, int, int, int *, GlobalLU_t *);
extern int     print_double_vec(char *, int, double *);
extern void    check_tempv(int, double *);

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_dSP_DEFS */

