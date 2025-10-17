/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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
#ifndef __SUPERLU_sSP_DEFS /* allow multiple inclusions */
#define __SUPERLU_sSP_DEFS

/*
 * File name:		ssp_defs.h
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
    float  *lusup;   /* L supernodes */
    int     *xlusup;
    float  *ucol;    /* U columns */
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
sgssv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
sgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, float *, float *, SuperMatrix *, SuperMatrix *,
       void *, int, SuperMatrix *, SuperMatrix *,
       float *, float *, float *, float *,
       mem_usage_t *, SuperLUStat_t *, int *);
    /* ILU */
extern void
sgsisv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
sgsisx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, float *, float *, SuperMatrix *, SuperMatrix *,
       void *, int, SuperMatrix *, SuperMatrix *, float *, float *,
       mem_usage_t *, SuperLUStat_t *, int *);


/*! \brief Supernodal LU factor related */
extern void
sCreate_CompCol_Matrix(SuperMatrix *, int, int, int, float *,
		       int *, int *, Stype_t, Dtype_t, Mtype_t);
extern void
sCreate_CompRow_Matrix(SuperMatrix *, int, int, int, float *,
		       int *, int *, Stype_t, Dtype_t, Mtype_t);
extern void
sCopy_CompCol_Matrix(SuperMatrix *, SuperMatrix *);
extern void
sCreate_Dense_Matrix(SuperMatrix *, int, int, float *, int,
		     Stype_t, Dtype_t, Mtype_t);
extern void
sCreate_SuperNode_Matrix(SuperMatrix *, int, int, int, float *, 
		         int *, int *, int *, int *, int *,
			 Stype_t, Dtype_t, Mtype_t);
extern void
sCopy_Dense_Matrix(int, int, float *, int, float *, int);

extern void    countnz (const int, int *, int *, int *, GlobalLU_t *);
extern void    ilu_countnz (const int, int *, int *, GlobalLU_t *);
extern void    fixupL (const int, const int *, GlobalLU_t *);

extern void    sallocateA (int, int, float **, int **, int **);
extern void    sgstrf (superlu_options_t*, SuperMatrix*,
                       int, int, int*, void *, int, int *, int *, 
                       SuperMatrix *, SuperMatrix *, SuperLUStat_t*, int *);
extern int     ssnode_dfs (const int, const int, const int *, const int *,
			     const int *, int *, int *, GlobalLU_t *);
extern int     ssnode_bmod (const int, const int, const int, float *,
                              float *, GlobalLU_t *, SuperLUStat_t*);
extern void    spanel_dfs (const int, const int, const int, SuperMatrix *,
			   int *, int *, float *, int *, int *, int *,
			   int *, int *, int *, int *, GlobalLU_t *);
extern void    spanel_bmod (const int, const int, const int, const int,
                           float *, float *, int *, int *,
			   GlobalLU_t *, SuperLUStat_t*);
extern int     scolumn_dfs (const int, const int, int *, int *, int *, int *,
			   int *, int *, int *, int *, int *, GlobalLU_t *);
extern int     scolumn_bmod (const int, const int, float *,
			   float *, int *, int *, int,
                           GlobalLU_t *, SuperLUStat_t*);
extern int     scopy_to_ucol (int, int, int *, int *, int *,
                              float *, GlobalLU_t *);         
extern int     spivotL (const int, const double, int *, int *, 
                         int *, int *, int *, GlobalLU_t *, SuperLUStat_t*);
extern void    spruneL (const int, const int *, const int, const int,
			  const int *, const int *, int *, GlobalLU_t *);
extern void    sreadmt (int *, int *, int *, float **, int **, int **);
extern void    sGenXtrue (int, int, float *, int);
extern void    sFillRHS (trans_t, int, float *, int, SuperMatrix *,
			  SuperMatrix *);
extern void    sgstrs (trans_t, SuperMatrix *, SuperMatrix *, int *, int *,
                        SuperMatrix *, SuperLUStat_t*, int *);
/* ILU */
extern void    sgsitrf (superlu_options_t*, SuperMatrix*, int, int, int*,
		        void *, int, int *, int *, SuperMatrix *, SuperMatrix *,
                        SuperLUStat_t*, int *);
extern int     sldperm(int, int, int, int [], int [], float [],
                        int [],	float [], float []);
extern int     ilu_ssnode_dfs (const int, const int, const int *, const int *,
			       const int *, int *, GlobalLU_t *);
extern void    ilu_spanel_dfs (const int, const int, const int, SuperMatrix *,
			       int *, int *, float *, float *, int *, int *,
			       int *, int *, int *, int *, GlobalLU_t *);
extern int     ilu_scolumn_dfs (const int, const int, int *, int *, int *,
				int *, int *, int *, int *, int *,
				GlobalLU_t *);
extern int     ilu_scopy_to_ucol (int, int, int *, int *, int *,
                                  float *, int, milu_t, double, int,
                                  float *, int *, GlobalLU_t *, float *);
extern int     ilu_spivotL (const int, const double, int *, int *, int, int *,
			    int *, int *, int *, double, milu_t,
                            float, GlobalLU_t *, SuperLUStat_t*);
extern int     ilu_sdrop_row (superlu_options_t *, int, int, double,
                              int, int *, double *, GlobalLU_t *, 
                              float *, float *, int);


/*! \brief Driver related */

extern void    sgsequ (SuperMatrix *, float *, float *, float *,
			float *, float *, int *);
extern void    slaqgs (SuperMatrix *, float *, float *, float,
                        float, float, char *);
extern void    sgscon (char *, SuperMatrix *, SuperMatrix *, 
		         float, float *, SuperLUStat_t*, int *);
extern float   sPivotGrowth(int, SuperMatrix *, int *, 
                            SuperMatrix *, SuperMatrix *);
extern void    sgsrfs (trans_t, SuperMatrix *, SuperMatrix *,
                       SuperMatrix *, int *, int *, char *, float *, 
                       float *, SuperMatrix *, SuperMatrix *,
                       float *, float *, SuperLUStat_t*, int *);

extern int     sp_strsv (char *, char *, char *, SuperMatrix *,
			SuperMatrix *, float *, SuperLUStat_t*, int *);
extern int     sp_sgemv (char *, float, SuperMatrix *, float *,
			int, float, float *, int);

extern int     sp_sgemm (char *, char *, int, int, int, float,
			SuperMatrix *, float *, int, float, 
			float *, int);
extern         float slamch_(char *);


/*! \brief Memory-related */
extern int     sLUMemInit (fact_t, void *, int, int, int, int, int,
                            float, SuperMatrix *, SuperMatrix *,
                            GlobalLU_t *, int **, float **);
extern void    sSetRWork (int, int, float *, float **, float **);
extern void    sLUWorkFree (int *, float *, GlobalLU_t *);
extern int     sLUMemXpand (int, int, MemType, int *, GlobalLU_t *);

extern float  *floatMalloc(int);
extern float  *floatCalloc(int);
extern int     smemory_usage(const int, const int, const int, const int);
extern int     sQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
extern int     ilu_sQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);

/*! \brief Auxiliary routines */
extern void    sreadhb(int *, int *, int *, float **, int **, int **);
extern void    sreadrb(int *, int *, int *, float **, int **, int **);
extern void    sreadtriple(int *, int *, int *, float **, int **, int **);
extern void    sCompRow_to_CompCol(int, int, int, float*, int*, int*,
		                   float **, int **, int **);
extern void    sfill (float *, int, float);
extern void    sinf_norm_error (int, SuperMatrix *, float *);
extern void    PrintPerf (SuperMatrix *, SuperMatrix *, mem_usage_t *,
			 float, float, float *, float *, char *);
extern float  sqselect(int, float *, int);


/*! \brief Routines for debugging */
extern void    sPrint_CompCol_Matrix(char *, SuperMatrix *);
extern void    sPrint_SuperNode_Matrix(char *, SuperMatrix *);
extern void    sPrint_Dense_Matrix(char *, SuperMatrix *);
extern void    sprint_lu_col(char *, int, int, int *, GlobalLU_t *);
extern int     print_double_vec(char *, int, double *);
extern void    check_tempv(int, float *);

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_sSP_DEFS */

