/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
#ifndef	_ITRACE_H
#define _ITRACE_H

/*
 *  Trace function prototypes
 */
#include "trace/proto.h"

extern int ODBCSharedTraceFlag;


/*
 *  Usefull macros
 */
#ifdef NO_TRACING
#define TRACE(X)
#else
#define TRACE(X)	if (ODBCSharedTraceFlag) X
#endif

#define TRACE_ENTER	0, retcode
#define TRACE_LEAVE	1, retcode

#define CALL_DRIVER(hdbc, errHandle, ret, proc, plist) \
    {\
	DBC_t *	t_pdbc = (DBC_t *)(hdbc);\
	ENV_t * t_penv = (ENV_t *)(t_pdbc->henv);\
\
	if (!t_penv->thread_safe) MUTEX_LOCK (t_penv->drv_lock); \
\
	ret = proc plist; \
	if (errHandle) ((GENV_t *)(errHandle))->rc = ret; \
\
	if (!t_penv->thread_safe) MUTEX_UNLOCK (t_penv->drv_lock); \
    }


#define CALL_UDRIVER(hdbc, errHandle, retcode, hproc, unicode_driver, procid, plist) \
    { \
	if (unicode_driver) \
	{ \
	    /* SQL_XXX_W */ \
	    hproc = _iodbcdm_getproc (hdbc, procid ## W); \
	} \
	else \
	{ \
	    /* SQL_XXX */   \
	    /* SQL_XXX_A */ \
	    hproc = _iodbcdm_getproc (hdbc, procid); \
	    if (hproc == SQL_NULL_HPROC) \
	        hproc = _iodbcdm_getproc (hdbc, procid ## A); \
	    } \
        if (hproc != SQL_NULL_HPROC) \
	      { \
	    CALL_DRIVER (hdbc, errHandle, retcode, hproc, plist) \
	} \
    }

#endif


#define GET_HPROC(hdbc, hproc, procid) \
    { \
      /* SQL_XXX */   \
      /* SQL_XXX_A */ \
      hproc = _iodbcdm_getproc (hdbc, procid); \
      if (hproc == SQL_NULL_HPROC) \
        hproc = _iodbcdm_getproc (hdbc, procid ## A); \
    }


#define GET_UHPROC(hdbc, hproc, procid, unicode_driver) \
    { \
      if (unicode_driver) \
        { \
	  /* SQL_XXX_W */ \
	  hproc = _iodbcdm_getproc (hdbc, procid ## W); \
        } \
      else \
        { \
          /* SQL_XXX */   \
          /* SQL_XXX_A */ \
          hproc = _iodbcdm_getproc (hdbc, procid); \
          if (hproc == SQL_NULL_HPROC) \
            hproc = _iodbcdm_getproc (hdbc, procid ## A); \
        } \
    }

