/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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
#ifndef __SUPERLU_zSP_DEFS /* allow multiple inclusions */
#define __SUPERLU_zSP_DEFS

/*
 * File name:		zsp_defs.h
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
#include "slu_dcomplex.h"



typedef struct {
    int     *xsup;    /* supernode and column mapping */
    int     *supno;   
    int     *lsub;    /* compressed L subscripts */
    int	    *xlsub;
    doublecomplex  *lusup;   /* L supernodes */
    int     *xlusup;
    doublecomplex  *ucol;    /* U columns */
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
zgssv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
zgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, double *, double *, SuperMatrix *, SuperMatrix *,
       void *, int, SuperMatrix *, SuperMatrix *,
       double *, double *, double *, double *,
       mem_usage_t *, SuperLUStat_t *, int *);
    /* ILU */
extern void
zgsisv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
zgsisx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, double *, double *, SuperMatrix *, SuperMatrix *,
       void *, int, SuperMatrix *, SuperMatrix *, double *, double *,
       mem_usage_t *, SuperLUStat_t *, int *);


/*! \brief Supernodal LU factor related */
extern void
zCreate_CompCol_Matrix(SuperMatrix *, int, int, int, doublecomplex *,
		       int *, int *, Stype_t, Dtype_t, Mtype_t);
extern void
zCreate_CompRow_Matrix(SuperMatrix *, int, int, int, doublecomplex *,
		       int *, int *, Stype_t, Dtype_t, Mtype_t);
extern void
zCopy_CompCol_Matrix(SuperMatrix *, SuperMatrix *);
extern void
zCreate_Dense_Matrix(SuperMatrix *, int, int, doublecomplex *, int,
		     Stype_t, Dtype_t, Mtype_t);
extern void
zCreate_SuperNode_Matrix(SuperMatrix *, int, int, int, doublecomplex *, 
		         int *, int *, int *, int *, int *,
			 Stype_t, Dtype_t, Mtype_t);
extern void
zCopy_Dense_Matrix(int, int, doublecomplex *, int, doublecomplex *, int);

extern void    countnz (const int, int *, int *, int *, GlobalLU_t *);
extern void    ilu_countnz (const int, int *, int *, GlobalLU_t *);
extern void    fixupL (const int, const int *, GlobalLU_t *);

extern void    zallocateA (int, int, doublecomplex **, int **, int **);
extern void    zgstrf (superlu_options_t*, SuperMatrix*,
                       int, int, int*, void *, int, int *, int *, 
                       SuperMatrix *, SuperMatrix *, SuperLUStat_t*, int *);
extern int     zsnode_dfs (const int, const int, const int *, const int *,
			     const int *, int *, int *, GlobalLU_t *);
extern int     zsnode_bmod (const int, const int, const int, doublecomplex *,
                              doublecomplex *, GlobalLU_t *, SuperLUStat_t*);
extern void    zpanel_dfs (const int, const int, const int, SuperMatrix *,
			   int *, int *, doublecomplex *, int *, int *, int *,
			   int *, int *, int *, int *, GlobalLU_t *);
extern void    zpanel_bmod (const int, const int, const int, const int,
                           doublecomplex *, doublecomplex *, int *, int *,
			   GlobalLU_t *, SuperLUStat_t*);
extern int     zcolumn_dfs (const int, const int, int *, int *, int *, int *,
			   int *, int *, int *, int *, int *, GlobalLU_t *);
extern int     zcolumn_bmod (const int, const int, doublecomplex *,
			   doublecomplex *, int *, int *, int,
                           GlobalLU_t *, SuperLUStat_t*);
extern int     zcopy_to_ucol (int, int, int *, int *, int *,
                              doublecomplex *, GlobalLU_t *);         
extern int     zpivotL (const int, const double, int *, int *, 
                         int *, int *, int *, GlobalLU_t *, SuperLUStat_t*);
extern void    zpruneL (const int, const int *, const int, const int,
			  const int *, const int *, int *, GlobalLU_t *);
extern void    zreadmt (int *, int *, int *, doublecomplex **, int **, int **);
extern void    zGenXtrue (int, int, doublecomplex *, int);
extern void    zFillRHS (trans_t, int, doublecomplex *, int, SuperMatrix *,
			  SuperMatrix *);
extern void    zgstrs (trans_t, SuperMatrix *, SuperMatrix *, int *, int *,
                        SuperMatrix *, SuperLUStat_t*, int *);
/* ILU */
extern void    zgsitrf (superlu_options_t*, SuperMatrix*, int, int, int*,
		        void *, int, int *, int *, SuperMatrix *, SuperMatrix *,
                        SuperLUStat_t*, int *);
extern int     zldperm(int, int, int, int [], int [], doublecomplex [],
                        int [],	double [], double []);
extern int     ilu_zsnode_dfs (const int, const int, const int *, const int *,
			       const int *, int *, GlobalLU_t *);
extern void    ilu_zpanel_dfs (const int, const int, const int, SuperMatrix *,
			       int *, int *, doublecomplex *, double *, int *, int *,
			       int *, int *, int *, int *, GlobalLU_t *);
extern int     ilu_zcolumn_dfs (const int, const int, int *, int *, int *,
				int *, int *, int *, int *, int *,
				GlobalLU_t *);
extern int     ilu_zcopy_to_ucol (int, int, int *, int *, int *,
                                  doublecomplex *, int, milu_t, double, int,
                                  doublecomplex *, int *, GlobalLU_t *, double *);
extern int     ilu_zpivotL (const int, const double, int *, int *, int, int *,
			    int *, int *, int *, double, milu_t,
                            doublecomplex, GlobalLU_t *, SuperLUStat_t*);
extern int     ilu_zdrop_row (superlu_options_t *, int, int, double,
                              int, int *, double *, GlobalLU_t *, 
                              double *, double *, int);


/*! \brief Driver related */

extern void    zgsequ (SuperMatrix *, double *, double *, double *,
			double *, double *, int *);
extern void    zlaqgs (SuperMatrix *, double *, double *, double,
                        double, double, char *);
extern void    zgscon (char *, SuperMatrix *, SuperMatrix *, 
		         double, double *, SuperLUStat_t*, int *);
extern double   zPivotGrowth(int, SuperMatrix *, int *, 
                            SuperMatrix *, SuperMatrix *);
extern void    zgsrfs (trans_t, SuperMatrix *, SuperMatrix *,
                       SuperMatrix *, int *, int *, char *, double *, 
                       double *, SuperMatrix *, SuperMatrix *,
                       double *, double *, SuperLUStat_t*, int *);

extern int     sp_ztrsv (char *, char *, char *, SuperMatrix *,
			SuperMatrix *, doublecomplex *, SuperLUStat_t*, int *);
extern int     sp_zgemv (char *, doublecomplex, SuperMatrix *, doublecomplex *,
			int, doublecomplex, doublecomplex *, int);

extern int     sp_zgemm (char *, char *, int, int, int, doublecomplex,
			SuperMatrix *, doublecomplex *, int, doublecomplex, 
			doublecomplex *, int);
extern         double dlamch_(char *);


/*! \brief Memory-related */
extern int     zLUMemInit (fact_t, void *, int, int, int, int, int,
                            double, SuperMatrix *, SuperMatrix *,
                            GlobalLU_t *, int **, doublecomplex **);
extern void    zSetRWork (int, int, doublecomplex *, doublecomplex **, doublecomplex **);
extern void    zLUWorkFree (int *, doublecomplex *, GlobalLU_t *);
extern int     zLUMemXpand (int, int, MemType, int *, GlobalLU_t *);

extern doublecomplex  *doublecomplexMalloc(int);
extern doublecomplex  *doublecomplexCalloc(int);
extern double  *doubleMalloc(int);
extern double  *doubleCalloc(int);
extern int     zmemory_usage(const int, const int, const int, const int);
extern int     zQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
extern int     ilu_zQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);

/*! \brief Auxiliary routines */
extern void    zreadhb(int *, int *, int *, doublecomplex **, int **, int **);
extern void    zreadrb(int *, int *, int *, doublecomplex **, int **, int **);
extern void    zreadtriple(int *, int *, int *, doublecomplex **, int **, int **);
extern void    zCompRow_to_CompCol(int, int, int, doublecomplex*, int*, int*,
		                   doublecomplex **, int **, int **);
extern void    zfill (doublecomplex *, int, doublecomplex);
extern void    zinf_norm_error (int, SuperMatrix *, doublecomplex *);
extern void    PrintPerf (SuperMatrix *, SuperMatrix *, mem_usage_t *,
			 doublecomplex, doublecomplex, doublecomplex *, doublecomplex *, char *);
extern double  dqselect(int, double *, int);


/*! \brief Routines for debugging */
extern void    zPrint_CompCol_Matrix(char *, SuperMatrix *);
extern void    zPrint_SuperNode_Matrix(char *, SuperMatrix *);
extern void    zPrint_Dense_Matrix(char *, SuperMatrix *);
extern void    zprint_lu_col(char *, int, int, int *, GlobalLU_t *);
extern int     print_double_vec(char *, int, double *);
extern void    check_tempv(int, doublecomplex *);

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_zSP_DEFS */

