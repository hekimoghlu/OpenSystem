/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
#include <iodbc.h>

#include <assert.h>
#include <sql.h>
#include <sqlext.h>

#include <odbcinst.h>

#include <dlproc.h>

#include <herr.h>
#include <henv.h>

#include <itrace.h>

/*
 *  Use static initializer where possible
 */

#if defined (PTHREAD_MUTEX_INITIALIZER)
SPINLOCK_DECLARE (iodbcdm_global_lock) = PTHREAD_MUTEX_INITIALIZER;
#else
SPINLOCK_DECLARE (iodbcdm_global_lock);
#endif

static int _iodbcdm_initialized = 0;


static void
_iodbcdm_env_settracing (GENV_t *genv)
{
  char buf[1024];

  genv = genv; /*UNUSED*/

  /*
   *  Check TraceFile keyword
   */
  SQLSetConfigMode (ODBC_BOTH_DSN);
  if( SQLGetPrivateProfileString ("ODBC", "TraceFile", "", buf, sizeof(buf) / sizeof(SQLTCHAR), "odbc.ini") == 0 || !buf[0])
    STRCPY (buf, SQL_OPT_TRACE_FILE_DEFAULT);
  trace_set_filename (buf);

  /*
   *  Check Trace keyword
   */
  SQLSetConfigMode (ODBC_BOTH_DSN);
  if ( SQLGetPrivateProfileString ("ODBC", "Trace", "", buf, sizeof(buf) / sizeof(SQLTCHAR), "odbc.ini") &&
      (STRCASEEQ (buf, "on") || STRCASEEQ (buf, "yes")
   || STRCASEEQ (buf, "1")))
    trace_start ();

  return;
}

unsigned long _iodbc_env_counter = 0;

SQLRETURN 
SQLAllocEnv_Internal (SQLHENV * phenv, int odbc_ver)
{
  GENV (genv, NULL);
  int retcode = SQL_SUCCESS;

  genv = (GENV_t *) MEM_ALLOC (sizeof (GENV_t));

  if (genv == NULL)
    {
      *phenv = SQL_NULL_HENV;

      return SQL_ERROR;
    }
  genv->rc = 0;

  /*
   *  Initialize this handle
   */
  genv->type = SQL_HANDLE_ENV;
  genv->henv = SQL_NULL_HENV;	/* driver's env list */
  genv->hdbc = SQL_NULL_HDBC;	/* driver's dbc list */
  genv->herr = SQL_NULL_HERR;	/* err list          */
#if (ODBCVER >= 0x300)
  genv->odbc_ver = odbc_ver;
  genv->connection_pooling = _iodbcdm_attr_connection_pooling;
  genv->cp_match = SQL_CP_MATCH_DEFAULT;
  genv->pdbc_pool = NULL;
#endif
  genv->err_rec = 0;

  *phenv = (SQLHENV) genv;

  /*
   * Initialize tracing 
   */
  if (++_iodbc_env_counter == 1)
    _iodbcdm_env_settracing (genv);

  return retcode;
}


SQLRETURN SQL_API
SQLAllocEnv (SQLHENV * phenv)
{
  GENV (genv, NULL);
  int retcode = SQL_SUCCESS;

  /* 
   *  One time initialization
   */
  Init_iODBC();

  ODBC_LOCK ();
  retcode = SQLAllocEnv_Internal (phenv, SQL_OV_ODBC2);

  genv = (GENV_t *) *phenv;

  /*
   * Start tracing
   */
  TRACE (trace_SQLAllocEnv (TRACE_ENTER, phenv));
  TRACE (trace_SQLAllocEnv (TRACE_LEAVE, phenv));

  ODBC_UNLOCK ();

  return retcode;
}


extern void _iodbcdm_pool_drop_conn (HDBC hdbc, HDBC hdbc_prev);

SQLRETURN
SQLFreeEnv_Internal (SQLHENV henv)
{
  GENV (genv, henv);

  if (!IS_VALID_HENV (genv))
    {
      return SQL_INVALID_HANDLE;
    }
  CLEAR_ERRORS (genv);

  if (genv->hdbc != SQL_NULL_HDBC)
    {
      PUSHSQLERR (genv->herr, en_S1010);

      return SQL_ERROR;
    }

#if (ODBCVER >= 0x300)
  /* Drop connections from the pool */
  while (genv->pdbc_pool != NULL)
    _iodbcdm_pool_drop_conn (genv->pdbc_pool, NULL);
#endif

  /*
   *  Invalidate this handle
   */
  genv->type = 0;

  return SQL_SUCCESS;
}


SQLRETURN SQL_API
SQLFreeEnv (SQLHENV henv)
{
  GENV (genv, henv);
  int retcode = SQL_SUCCESS;

  ODBC_LOCK ();

  TRACE (trace_SQLFreeEnv (TRACE_ENTER, henv));

  retcode = SQLFreeEnv_Internal (henv);

  TRACE (trace_SQLFreeEnv (TRACE_LEAVE, henv));

  MEM_FREE (genv);

  /*
   *  Close trace after last environment handle is freed
   */
  if (--_iodbc_env_counter == 0)
    trace_stop();

  ODBC_UNLOCK ();

  return retcode;
}


/*
 *  Initialize the system and let everyone wait until we have done so
 *  properly
 */
void
Init_iODBC (void)
{
#if !defined (PTHREAD_MUTEX_INITIALIZER) || defined (WINDOWS)
  SPINLOCK_INIT (iodbcdm_global_lock);
#endif

  SPINLOCK_LOCK (iodbcdm_global_lock);
  if (!_iodbcdm_initialized)
    {
      /*
       *  OK, now flag we are not callable anymore
       */
      _iodbcdm_initialized = 1;

      /*
       *  Other one time initializations can be performed here
       */
    }
  SPINLOCK_UNLOCK (iodbcdm_global_lock);

  return;
}


void 
Done_iODBC(void)
{
#if !defined (PTHREAD_MUTEX_INITIALIZER) || defined (WINDOWS)
    SPINLOCK_DONE (iodbcdm_global_lock);
#endif
}


/*
 *  DLL Entry points for Windows
 */
#if defined (WINDOWS)
static int
DLLInit (HINSTANCE hModule)
{
  Init_iODBC ();

  return TRUE;
}


static void
DLLExit (void)
{
  Done_iODBC ();
}


#pragma argused
BOOL WINAPI
DllMain (HINSTANCE hModule, DWORD fdReason, LPVOID lpvReserved)
{
  switch (fdReason)
    {
    case DLL_PROCESS_ATTACH:
      if (!DLLInit (hModule))
	return FALSE;
      break;
    case DLL_PROCESS_DETACH:
      DLLExit ();
    }
  return TRUE;
}
#endif
