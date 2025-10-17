/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#ifndef	_HDBC_H
#define	_HDBC_H

#if (ODBCVER >= 0x0300)
#include <hdesc.h>
#endif

typedef struct _drvopt
  {
    SQLUSMALLINT Option;
    SQLULEN Param;
    SQLCHAR waMode;

    struct _drvopt *next;
  } 
DRVOPT;

typedef struct DBC
  {
    int type;			/* must be 1st field */
    HERR herr;
    SQLRETURN rc;

    struct DBC * next;

    HENV genv;			/* back point to global env object */

    HDBC dhdbc;			/* driver's private dbc */
    HENV henv;			/* back point to instant env object */
    HSTMT hstmt;		/* list of statement object handle(s) */
#if (ODBCVER >= 0x300)
    HDESC hdesc;    		/* list of connection descriptors */

    struct DBC * cp_pdbc;	/* pooled connection */
    BOOL cp_in_use;		/* connection in pool is in use */
    time_t cp_timeout;		/* CPTimeout parameter */
    time_t cp_expiry_time;	/* expiration time (abs time value) */
    time_t cp_retry_wait;	/* timeout before retry (abs time value) */
    char *cp_probe;		/* CPProbe -- probe SQL statement */
    char *cp_dsn;
    char *cp_uid;
    char *cp_pwd;
    char *cp_connstr;
#endif    

    int state;

    /* options */
    UDWORD access_mode;
    UDWORD autocommit;

    UDWORD login_timeout;
    UDWORD odbc_cursors;
    UDWORD packet_size;
    UDWORD quiet_mode;
    UDWORD txn_isolation;
    SWORD cb_commit;
    SWORD cb_rollback;

    wchar_t * current_qualifier;
    char current_qualifier_WA;

    SWORD dbc_cip;			/* Call in Progess flag */

    DRVOPT *drvopt;			/* Driver specific connect options */
    SQLSMALLINT err_rec;
  }
DBC_t;


#define IS_VALID_HDBC(x) \
	((x) != SQL_NULL_HDBC && ((DBC_t *)(x))->type == SQL_HANDLE_DBC)


#define ENTER_HDBC(hdbc, holdlock, trace) \
	CONN(pdbc, hdbc); \
        SQLRETURN retcode = SQL_SUCCESS; \
        ODBC_LOCK();\
	TRACE(trace); \
    	if (!IS_VALID_HDBC (pdbc)) \
	  { \
	    retcode = SQL_INVALID_HANDLE; \
	    goto done; \
	  } \
	else if (pdbc->dbc_cip) \
          { \
	    PUSHSQLERR (pdbc->herr, en_S1010); \
	    retcode = SQL_ERROR; \
	    goto done; \
	  } \
	pdbc->dbc_cip = 1; \
	CLEAR_ERRORS (pdbc); \
	if (!holdlock) \
	  ODBC_UNLOCK()


#define LEAVE_HDBC(hdbc, holdlock, trace) \
	if (!holdlock) \
	  ODBC_LOCK (); \
	pdbc->dbc_cip = 0; \
    done: \
    	TRACE(trace); \
	ODBC_UNLOCK (); \
	return (retcode)


/* 
 * Note:
 *  - ODBC applications can see address of driver manager's 
 *    connection object, i.e connection handle -- a void pointer, 
 *    but not detail of it. ODBC applications can neither see 
 *    detail driver's connection object nor its address.
 *
 *  - ODBC driver manager knows its own connection objects and
 *    exposes their address to an ODBC application. Driver manager
 *    also knows address of driver's connection objects and keeps
 *    it via dhdbc field in driver manager's connection object.
 * 
 *  - ODBC driver exposes address of its own connection object to
 *    driver manager without detail.
 *
 *  - Applications can get driver's connection object handle by
 *    SQLGetInfo() with fInfoType equals to SQL_DRIVER_HDBC.
 */

enum
  {
    en_dbc_allocated,
    en_dbc_needdata,
    en_dbc_connected,
    en_dbc_hstmt
  };


/*
 *  Internal prototypes 
 */
SQLRETURN SQL_API _iodbcdm_SetConnectOption (
    SQLHDBC hdbc,
    SQLUSMALLINT fOption, 
    SQLULEN vParam,
    SQLCHAR waMode);
SQLRETURN SQL_API _iodbcdm_GetConnectOption (
    SQLHDBC hdbc,
    SQLUSMALLINT fOption, 
    SQLPOINTER pvParam,
    SQLCHAR waMode);
#endif
