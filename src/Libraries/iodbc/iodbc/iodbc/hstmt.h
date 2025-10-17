/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#ifndef	_HSTMT_H
#define	_HSTMT_H

typedef struct PARAM
  {
    void *data;
    int   length;
  }
PARAM_t;

#define STMT_PARAMS_MAX      8


/*
 *  Binding parameter from SQLBindCol
 */
typedef struct BIND {
  UWORD		 bn_col;	  /* Column # */
  SWORD		 bn_type;	  /* ODBC C data type */
  void *	 bn_data;	  /* Pointer to data */
  SDWORD	 bn_size;	  /* Size of data area */
  SQLLEN	*bn_pInd;	  /* Holds SQL_NULL_DATA | 0. 
                                   * And length of returned char/bin data 
				   */
} BIND_t;

typedef struct SBLST	TBLST, *PBLST;
/*
 *  Binding cell on the linked list
 */
struct SBLST {
  PBLST		 bl_nextBind;	/* Next binding */
  BIND_t	 bl_bind;	/* Binding information */
};


typedef struct STMT
  {
    int type;			/* must be 1st field */
    HERR herr;
    SQLRETURN rc;		/* Return code of last function */

    struct STMT *next;

    HDBC hdbc;			/* back point to connection object */

    HSTMT dhstmt;		/* driver's stmt handle */

    int state;
    int cursor_state;
    int prep_state;
    int asyn_on;		/* async executing which odbc call */
    int need_on;		/* which call return SQL_NEED_DATA */

    int stmt_cip;		/* Call in progress on this handle */

    SQLUINTEGER rowset_size;
    SQLUINTEGER bind_type;

#if (ODBCVER >= 0x0300)
    DESC_t * imp_desc[4];
    DESC_t * desc[4];
    SQLUINTEGER row_array_size;
    SQLPOINTER fetch_bookmark_ptr, params_processed_ptr;
    SQLUINTEGER paramset_size;
    SQLPOINTER row_status_ptr;
    SQLPOINTER rows_fetched_ptr;
    SQLUSMALLINT row_status_allocated;
#endif

    SQLSMALLINT err_rec;

    PARAM_t params[STMT_PARAMS_MAX]; /* for a conversion parameters ansi<=>unicode*/
    int     params_inserted;

    PBLST   st_pbinding;	/* API user bindings from SQLBindCol */
  }
STMT_t;


#define IS_VALID_HSTMT(x) \
	((x) != SQL_NULL_HSTMT && \
	 ((STMT_t *)(x))->type == SQL_HANDLE_STMT && \
	 ((STMT_t *)(x))->hdbc != SQL_NULL_HDBC)


#define ENTER_STMT(hstmt, trace) \
	STMT (pstmt, hstmt); \
	SQLRETURN retcode = SQL_SUCCESS; \
        ODBC_LOCK(); \
	TRACE (trace); \
    	if (!IS_VALID_HSTMT (pstmt)) \
	  { \
	    retcode = SQL_INVALID_HANDLE; \
	    goto done; \
	  } \
	else if (pstmt->stmt_cip) \
          { \
	    PUSHSQLERR (pstmt->herr, en_S1010); \
	    retcode = SQL_ERROR; \
	    goto done; \
	  } \
	pstmt->stmt_cip = 1; \
	CLEAR_ERRORS (pstmt); \
	if (pstmt->asyn_on == en_NullProc && pstmt->params_inserted > 0) \
	  _iodbcdm_FreeStmtParams(pstmt); \
        ODBC_UNLOCK()
	

#define LEAVE_STMT(hstmt, trace) \
	ODBC_LOCK (); \
	pstmt->stmt_cip = 0; \
    done: \
    	TRACE(trace); \
	ODBC_UNLOCK (); \
	return (retcode)


enum
  {
    en_stmt_allocated = 0,
    en_stmt_prepared,
    en_stmt_executed_with_info,
    en_stmt_executed,
    en_stmt_cursoropen,
    en_stmt_fetched,
    en_stmt_xfetched,
    en_stmt_needdata,		/* not call SQLParamData() yet */
    en_stmt_mustput,		/* not call SQLPutData() yet */
    en_stmt_canput		/* SQLPutData() called */
  };				/* for statement handle state */

enum
  {
    en_stmt_cursor_no = 0,
    en_stmt_cursor_named,
    en_stmt_cursor_opened,
    en_stmt_cursor_fetched,
    en_stmt_cursor_xfetched
  };				/* for statement cursor state */


/*
 *  Internal prototypes
 */
SQLRETURN _iodbcdm_dropstmt (HSTMT stmt);

void _iodbcdm_FreeStmtParams(STMT_t *pstmt);
void *_iodbcdm_alloc_param(STMT_t *pstmt, int i, int size);
wchar_t *_iodbcdm_conv_param_A2W(STMT_t *pstmt, int i, SQLCHAR *pData, int pDataLength);
char *_iodbcdm_conv_param_W2A(STMT_t *pstmt, int i, SQLWCHAR *pData, int pDataLength);
void _iodbcdm_ConvBindData (STMT_t *pstmt);
SQLRETURN _iodbcdm_BindColumn (STMT_t *pstmt, BIND_t *pbind);
int _iodbcdm_UnBindColumn (STMT_t *pstmt, BIND_t *pbind);
void _iodbcdm_RemoveBind (STMT_t *pstmt);
void _iodbcdm_do_cursoropen (STMT_t * pstmt);
SQLSMALLINT _iodbcdm_map_sql_type (int type, int odbcver);
SQLSMALLINT _iodbcdm_map_c_type (int type, int odbcver);


SQLRETURN SQL_API _iodbcdm_ExtendedFetch (
    SQLHSTMT		  hstmt,
    SQLUSMALLINT	  fFetchType,
    SQLLEN		  irow, 
    SQLULEN	 	* pcrow, 
    SQLUSMALLINT 	* rgfRowStatus);

SQLRETURN SQL_API _iodbcdm_SetPos (
    SQLHSTMT		  hstmt, 
    SQLSETPOSIROW	  irow, 
    SQLUSMALLINT	  fOption, 
    SQLUSMALLINT	  fLock);

SQLRETURN SQL_API _iodbcdm_NumResultCols (
    SQLHSTMT hstmt,
    SQLSMALLINT * pccol);

SQLRETURN SQLGetStmtOption_Internal (
  SQLHSTMT	hstmt, 
  SQLUSMALLINT	fOption, 
  SQLPOINTER	pvParam);
#endif
