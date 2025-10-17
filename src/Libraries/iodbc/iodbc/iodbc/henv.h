/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#ifndef	_HENV_H
#define	_HENV_H

#include <iodbc.h>
#include <dlproc.h>

#include <sql.h>
#include <sqlext.h>
#include <ithread.h>


enum odbcapi_t
  {
    en_NullProc = 0
#define FUNCDEF(A, B, C)	,B
#include "henv.ci"
#undef FUNCDEF
    , __LAST_API_FUNCTION__
  } ;


#if (ODBCVER >= 0x300)
/*
 * SQL_ATTR_CONNECTION_POOLING value
 */
extern SQLINTEGER _iodbcdm_attr_connection_pooling;
#endif


typedef struct
  {
    int type;			/* must be 1st field */
    HERR herr;			/* err list          */
    SQLRETURN rc;

    HENV henv;			/* driver's env list */
    HDBC hdbc;			/* driver's dbc list */
    int state;
#if (ODBCVER >= 0x300)
    SQLUINTEGER odbc_ver;    /* ODBC version of the application */

    SQLINTEGER connection_pooling; /* SQL_ATTR_CONNECTION_POOLING value at the
				      time of env creation */
    SQLINTEGER cp_match;	/* connection pool matching method */
    struct DBC *pdbc_pool;	/* connection pool */
#endif    

    SQLSMALLINT err_rec;
  }
GENV_t;


typedef struct
  {
    HENV next;			/* next attached env handle */
    int refcount;		/* Driver's bookkeeping reference count */
    HPROC dllproc_tab[__LAST_API_FUNCTION__];	/* driver api calls  */
    HENV dhenv;			/* driver env handle    */
    HDLL hdll;			/* driver share library handle */

    SWORD thread_safe;		/* Is the driver threadsafe? */
    SWORD unicode_driver;       /* Is the driver unicode? */
    MUTEX_DECLARE (drv_lock);	/* Used only when driver is not threadsafe */

#if (ODBCVER >= 0x300)
    SQLUINTEGER dodbc_ver;	/* driver's ODBC version */
#endif    
  }
ENV_t;


#define IS_VALID_HENV(x) \
	((x) != SQL_NULL_HENV && ((GENV_t *)(x))->type == SQL_HANDLE_ENV)


#define ENTER_HENV(henv, trace) \
	GENV (genv, henv); \
	SQLRETURN retcode = SQL_SUCCESS; \
	ODBC_LOCK (); \
	TRACE (trace); \
	if (!IS_VALID_HENV (henv)) \
	  { \
	    retcode = SQL_INVALID_HANDLE; \
	    goto done; \
	  } \
	CLEAR_ERRORS (genv)


#define LEAVE_HENV(henv, trace) \
    done: \
     	TRACE(trace); \
	ODBC_UNLOCK (); \
	return (retcode)
  

/*
 * Multi threading
 */
#if defined (IODBC_THREADING)
extern SPINLOCK_DECLARE(iodbcdm_global_lock);
#define ODBC_LOCK()	SPINLOCK_LOCK(iodbcdm_global_lock)
#define ODBC_UNLOCK()	SPINLOCK_UNLOCK(iodbcdm_global_lock)
#else
#define ODBC_LOCK()
#define ODBC_UNLOCK()
#endif

/*
 * Prototypes
 */
void Init_iODBC(void);
void Done_iODBC(void);



/* Note:
 *
 *  - ODBC applications only know about global environment handle, 
 *    a void pointer points to a GENV_t object. There is only one
 *    this object per process(however, to make the library reentrant,
 *    we still keep this object on heap). Applications only know 
 *    address of this object and needn't care about its detail.
 *
 *  - ODBC driver manager knows about instance environment handles,
 *    void pointers point to ENV_t objects. There are maybe more
 *    than one this kind of objects per process. However, multiple
 *    connections to a same data source(i.e. call same share library)
 *    will share one instance environment object.
 *
 *  - ODBC driver manager knows about their own environment handle,
 *    a void pointer point to a driver defined object. Every driver
 *    keeps one of its own environment object and driver manager
 *    keeps address of it by the 'dhenv' field in the instance
 *    environment object without care about its detail.
 *
 *  - Applications can get driver's environment object handle by
 *    SQLGetInfo() with fInfoType equals to SQL_DRIVER_HENV
 */
#endif
