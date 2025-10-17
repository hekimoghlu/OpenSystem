/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
#ifndef	_DLPROC_H
#define	_DLPROC_H

#include <dlf.h>

#if defined(_MAC) || defined (__cplusplus)
typedef SQLRETURN (* HPROC) (...);
#else
typedef SQLRETURN (* HPROC) ();
#endif

#ifdef	DLDAPI_SVR4_DLFCN
#include <dlfcn.h>
#endif

#ifdef DLDAPI_HP_SHL
#include <dl.h>
typedef shl_t HDLL;
#endif

#if defined(_BE)		|| \
    defined(_MAC)		|| \
    defined(_MACX)		|| \
    defined(DLDAPI_AIX_LOAD)	|| \
    defined(DLDAPI_DYLD)	|| \
    defined(DLDAPI_MACX)	|| \
    defined(DLDAPI_SVR4_DLFCN)	|| \
    defined(VMS)
typedef void *HDLL;
#endif


typedef struct _dl_s
{
  char		* path;
  HDLL		  dll;
  unsigned int    refcount;
  int 		  safe_unload;
  struct _dl_s	* next;
} dlproc_t;


/* dlproc.c */
HPROC _iodbcdm_getproc (HDBC hdbc, int idx);
HDLL _iodbcdm_dllopen (char *path);
HPROC _iodbcdm_dllproc (HDLL hdll, char *sym);
int _iodbcdm_dllclose (HDLL hdll);
char *_iodbcdm_dllerror (void);
void _iodbcdm_safe_unload (HDLL hdll);

#define	SQL_NULL_HDLL	((HDLL)NULL)
#define	SQL_NULL_HPROC	((HPROC)NULL)
#endif
