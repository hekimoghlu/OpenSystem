/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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

#include <sql.h>
#include <sqlext.h>
#include <sqlucode.h>

#include <dlproc.h>

#include <herr.h>
#include <henv.h>
#include <hdbc.h>
#include <hstmt.h>

#include <itrace.h>
#include <unicode.h>

SQLRETURN
SQLFetch_Internal (SQLHSTMT hstmt)
{
  STMT (pstmt, hstmt);
  HPROC hproc = SQL_NULL_HPROC;
  SQLRETURN retcode = SQL_SUCCESS;

  /* check state */
  if (pstmt->asyn_on == en_NullProc)
    {
      switch (pstmt->state)
	{
	case en_stmt_allocated:
	case en_stmt_prepared:
	case en_stmt_xfetched:
	case en_stmt_needdata:
	case en_stmt_mustput:
	case en_stmt_canput:
	  PUSHSQLERR (pstmt->herr, en_S1010);
	  return SQL_ERROR;

	case en_stmt_executed_with_info:
	  _iodbcdm_do_cursoropen (pstmt);
	  break;

	default:
	  break;
	}
    }
  else if (pstmt->asyn_on != en_Fetch)
    {
      PUSHSQLERR (pstmt->herr, en_S1010);
      return SQL_ERROR;
    }

#if (ODBCVER >= 0x0300)
  if (((ENV_t *) ((DBC_t *) pstmt->hdbc)->henv)->dodbc_ver ==  SQL_OV_ODBC2
      && ((GENV_t *) ((DBC_t *) pstmt->hdbc)->genv)->odbc_ver == SQL_OV_ODBC3)
    {				
	/* 
	 *  Try to map SQLFetch to SQLExtendedFetch for ODBC3 app calling 
	 *  ODBC2 driver 
         * 
	 *  The rows_status_ptr must not be null because the SQLExtendedFetch 
	 *  requires it 
	 */
      hproc = _iodbcdm_getproc (pstmt->hdbc, en_ExtendedFetch);

      if (hproc)
	{
	  CALL_DRIVER (pstmt->hdbc, pstmt, retcode, hproc,
	      (pstmt->dhstmt, SQL_FETCH_NEXT, 0, pstmt->rows_fetched_ptr,
		  pstmt->row_status_ptr));
	}
    }
#endif
  if (hproc == SQL_NULL_HPROC)
    {
      hproc = _iodbcdm_getproc (pstmt->hdbc, en_Fetch);

      if (hproc == SQL_NULL_HPROC)
	{
	  PUSHSQLERR (pstmt->herr, en_IM001);
	  return SQL_ERROR;
	}

      CALL_DRIVER (pstmt->hdbc, pstmt, retcode, hproc,
	  (pstmt->dhstmt));
    }

  /* state transition */
  if (pstmt->asyn_on == en_Fetch)
    {
      switch (retcode)
	{
	case SQL_SUCCESS:
	case SQL_SUCCESS_WITH_INFO:
	case SQL_NO_DATA_FOUND:
	case SQL_ERROR:
	  pstmt->asyn_on = en_NullProc;
	  break;

	case SQL_STILL_EXECUTING:
	default:
	  return retcode;
	}
    }

  switch (pstmt->state)
    {
    case en_stmt_cursoropen:
    case en_stmt_fetched:
      switch (retcode)
	{
	case SQL_SUCCESS:
	case SQL_SUCCESS_WITH_INFO:
	  pstmt->state = en_stmt_fetched;
	  pstmt->cursor_state = en_stmt_cursor_fetched;
	  break;

	case SQL_NO_DATA_FOUND:
	  if (pstmt->prep_state)
	    {
	      pstmt->state = en_stmt_fetched;
	    }
	  else
	    {
	      pstmt->state = en_stmt_allocated;
	    }
	  pstmt->cursor_state = en_stmt_cursor_no;
	  break;

	case SQL_STILL_EXECUTING:
	  pstmt->asyn_on = en_Fetch;
	  break;

	default:
	  break;
	}
      break;

    default:
      break;
    }

  return retcode;
}


SQLRETURN SQL_API
SQLFetch (SQLHSTMT hstmt)
{
  ENTER_STMT (hstmt,
    trace_SQLFetch (TRACE_ENTER, hstmt));

  retcode = SQLFetch_Internal (hstmt);

  if (SQL_SUCCEEDED (retcode))
    _iodbcdm_ConvBindData (pstmt);

  LEAVE_STMT (hstmt,
    trace_SQLFetch (TRACE_LEAVE, hstmt));
}


SQLRETURN SQL_API
_iodbcdm_ExtendedFetch (
    SQLHSTMT		  hstmt,
    SQLUSMALLINT	  fFetchType,
    SQLLEN		  irow, 
    SQLULEN	 	* pcrow, 
    SQLUSMALLINT 	* rgfRowStatus)
{
  STMT (pstmt, hstmt);
  HPROC hproc = SQL_NULL_HPROC;
  SQLRETURN retcode;

  /* check fetch type */
  if (fFetchType < SQL_FETCH_NEXT || fFetchType > SQL_FETCH_BOOKMARK)
    {
      /* Unlike MS driver manager(i.e. DM),
       * we don't check driver's ODBC version 
       * against SQL_FETCH_RESUME (only 1.0)
       * and SQL_FETCH_BOOKMARK (only 2.0).
       */
      PUSHSQLERR (pstmt->herr, en_S1106);

      return SQL_ERROR;
    }

  /* check state */
  if (pstmt->asyn_on == en_NullProc)
    {
      switch (pstmt->state)
	{
	case en_stmt_allocated:
	case en_stmt_prepared:
	case en_stmt_fetched:
	case en_stmt_needdata:
	case en_stmt_mustput:
	case en_stmt_canput:
	  PUSHSQLERR (pstmt->herr, en_S1010);
	  return SQL_ERROR;

	default:
	  break;
	}
    }
  else if (pstmt->asyn_on != en_ExtendedFetch)
    {
      PUSHSQLERR (pstmt->herr, en_S1010);
      return SQL_ERROR;
    }

  if (fFetchType == SQL_FETCH_NEXT ||
      fFetchType == SQL_FETCH_PRIOR ||
      fFetchType == SQL_FETCH_FIRST || fFetchType == SQL_FETCH_LAST)
    {
      irow = 0;
    }

  hproc = _iodbcdm_getproc (pstmt->hdbc, en_ExtendedFetch);

  if (hproc == SQL_NULL_HPROC)
    {
      PUSHSQLERR (pstmt->herr, en_IM001);

      return SQL_ERROR;
    }

  CALL_DRIVER (pstmt->hdbc, pstmt, retcode, hproc,
      (pstmt->dhstmt, fFetchType, irow, pcrow, rgfRowStatus));

  /* state transition */
  if (pstmt->asyn_on == en_ExtendedFetch)
    {
      switch (retcode)
	{
	case SQL_SUCCESS:
	case SQL_SUCCESS_WITH_INFO:
	case SQL_NO_DATA_FOUND:
	case SQL_ERROR:
	  pstmt->asyn_on = en_NullProc;
	  break;

	case SQL_STILL_EXECUTING:
	default:
	  return retcode;
	}
    }

  switch (pstmt->state)
    {
    case en_stmt_cursoropen:
    case en_stmt_xfetched:
      switch (retcode)
	{
	case SQL_SUCCESS:
	case SQL_SUCCESS_WITH_INFO:
	case SQL_NO_DATA_FOUND:
	  pstmt->state = en_stmt_xfetched;
	  pstmt->cursor_state = en_stmt_cursor_xfetched;
	  break;

	case SQL_STILL_EXECUTING:
	  pstmt->asyn_on = en_ExtendedFetch;
	  break;

	default:
	  break;
	}
      break;

    default:
      break;
    }

  return retcode;
}


SQLRETURN SQL_API
SQLExtendedFetch (
  SQLHSTMT		  hstmt, 
  SQLUSMALLINT		  fFetchType, 
  SQLLEN		  irow, 
  SQLULEN 		* pcrow, 
  SQLUSMALLINT 		* rgfRowStatus)
{
  ENTER_STMT (hstmt,
    trace_SQLExtendedFetch (TRACE_ENTER,
    	hstmt, fFetchType, irow, pcrow, rgfRowStatus));

  retcode =
      _iodbcdm_ExtendedFetch (hstmt, fFetchType, irow, pcrow, rgfRowStatus);

  if (SQL_SUCCEEDED (retcode))
    _iodbcdm_ConvBindData (pstmt);

  LEAVE_STMT (hstmt,
    trace_SQLExtendedFetch (TRACE_LEAVE,
    	hstmt, fFetchType, irow, pcrow, rgfRowStatus));
}


static SQLRETURN
SQLGetData_Internal (
  SQLHSTMT		  hstmt,
  SQLUSMALLINT		  icol,
  SQLSMALLINT		  fCType,
  SQLPOINTER		  rgbValue,
  SQLLEN		  cbValueMax,
  SQLLEN 		* pcbValue)
{
  STMT (pstmt, hstmt);
  CONN (pdbc, pstmt->hdbc);
  ENVR (penv, pdbc->henv);
  HPROC hproc;
  SQLRETURN retcode = SQL_SUCCESS;
  sqlstcode_t sqlstat = en_00000;
  SQLSMALLINT nCType;

  /* check argument */
  if (rgbValue == NULL)
    {
      sqlstat = en_S1009;
    }
  else if (cbValueMax < 0)
    {
      sqlstat = en_S1090;
    }
  else
    {
      switch (fCType)
	{
	case SQL_C_DEFAULT:
	case SQL_C_BINARY:
	case SQL_C_BIT:
	case SQL_C_CHAR:
	case SQL_C_DATE:
	case SQL_C_DOUBLE:
	case SQL_C_FLOAT:
	case SQL_C_LONG:
	case SQL_C_SHORT:
	case SQL_C_SLONG:
	case SQL_C_SSHORT:
	case SQL_C_STINYINT:
	case SQL_C_TIME:
	case SQL_C_TIMESTAMP:
	case SQL_C_TINYINT:
	case SQL_C_ULONG:
	case SQL_C_USHORT:
	case SQL_C_UTINYINT:
#if (ODBCVER >= 0x0300)
	case SQL_C_GUID:
	case SQL_C_INTERVAL_DAY:
	case SQL_C_INTERVAL_DAY_TO_HOUR:
	case SQL_C_INTERVAL_DAY_TO_MINUTE:
	case SQL_C_INTERVAL_DAY_TO_SECOND:
	case SQL_C_INTERVAL_HOUR:
	case SQL_C_INTERVAL_HOUR_TO_MINUTE:
	case SQL_C_INTERVAL_HOUR_TO_SECOND:
	case SQL_C_INTERVAL_MINUTE:
	case SQL_C_INTERVAL_MINUTE_TO_SECOND:
	case SQL_C_INTERVAL_MONTH:
	case SQL_C_INTERVAL_SECOND:
	case SQL_C_INTERVAL_YEAR:
	case SQL_C_INTERVAL_YEAR_TO_MONTH:
	case SQL_C_NUMERIC:
	case SQL_C_SBIGINT:
	case SQL_C_TYPE_DATE:
	case SQL_C_TYPE_TIME:
	case SQL_C_TYPE_TIMESTAMP:
	case SQL_C_UBIGINT:
	case SQL_C_WCHAR:
#endif
	  break;

	default:
	  sqlstat = en_S1003;
	  break;
	}
    }

  if (sqlstat != en_00000)
    {
      PUSHSQLERR (pstmt->herr, sqlstat);

      return SQL_ERROR;
    }

  /* check state */
  if (pstmt->asyn_on == en_NullProc)
    {
      switch (pstmt->state)
	{
	case en_stmt_allocated:
	case en_stmt_prepared:
	case en_stmt_needdata:
	case en_stmt_mustput:
	case en_stmt_canput:
	  sqlstat = en_S1010;
	  break;

	case en_stmt_executed_with_info:
	case en_stmt_executed:
	case en_stmt_cursoropen:
	  sqlstat = en_24000;
	  break;

	default:
	  break;
	}
    }
  else if (pstmt->asyn_on != en_GetData)
    {
      sqlstat = en_S1010;
    }

  if (sqlstat != en_00000)
    {
      PUSHSQLERR (pstmt->herr, sqlstat);

      return SQL_ERROR;
    }


  /*
   *  Convert C type to ODBC version of driver
   */
  nCType = _iodbcdm_map_c_type (fCType, penv->dodbc_ver);

  if (!penv->unicode_driver && nCType == SQL_C_WCHAR)
    {
      nCType = SQL_C_CHAR;
      cbValueMax /= sizeof(wchar_t);
    }


  /* call driver */
  hproc = _iodbcdm_getproc (pstmt->hdbc, en_GetData);

  if (hproc == SQL_NULL_HPROC)
    {
      PUSHSQLERR (pstmt->herr, en_IM001);
      return SQL_ERROR;
    }

  CALL_DRIVER (pstmt->hdbc, pstmt, retcode, hproc,
      (pstmt->dhstmt, icol, nCType, rgbValue, cbValueMax, pcbValue));

  /* state transition */
  if (pstmt->asyn_on == en_GetData)
    {
      switch (retcode)
	{
	case SQL_SUCCESS:
	case SQL_SUCCESS_WITH_INFO:
	case SQL_NO_DATA_FOUND:
	case SQL_ERROR:
	  pstmt->asyn_on = en_NullProc;
	  break;

	case SQL_STILL_EXECUTING:
	default:
	  return retcode;
	}
    }

  switch (pstmt->state)
    {
    case en_stmt_fetched:
    case en_stmt_xfetched:
      if (retcode == SQL_STILL_EXECUTING)
	{
	  pstmt->asyn_on = en_GetData;
	  break;
	}
      break;

    default:
      break;
    }

  if (!penv->unicode_driver && fCType == SQL_C_WCHAR)
    {
      wchar_t *buf = dm_SQL_A2W((SQLCHAR *) rgbValue, SQL_NTS);

      if (buf != NULL) 
        WCSCPY(rgbValue, buf);

      MEM_FREE(buf);
      if (pcbValue)
      	*pcbValue *= sizeof(wchar_t);
    }

  return retcode;
}


SQLRETURN SQL_API
SQLGetData (
    SQLHSTMT		  hstmt,
    SQLUSMALLINT	  icol,
    SQLSMALLINT		  fCType,
    SQLPOINTER		  rgbValue,
    SQLLEN		  cbValueMax,
    SQLLEN 		* pcbValue)
{
  ENTER_STMT (hstmt,
    trace_SQLGetData (TRACE_ENTER, 
    	hstmt, 
	icol, 
	fCType, 
    	rgbValue, cbValueMax, pcbValue));

  retcode = SQLGetData_Internal (
    	hstmt, 
	icol, 
	fCType, 
    	rgbValue, cbValueMax, pcbValue);

  LEAVE_STMT (hstmt,
    trace_SQLGetData (TRACE_LEAVE, 
    	hstmt, 
	icol, 
	fCType, 
    	rgbValue, cbValueMax, pcbValue));
}


static SQLRETURN
SQLMoreResults_Internal (SQLHSTMT hstmt)
{
  STMT (pstmt, hstmt);
  HPROC hproc;
  SQLRETURN retcode;

  /* check state */
  if (pstmt->asyn_on == en_NullProc)
    {
      switch (pstmt->state)
	{
#if 0
	case en_stmt_allocated:
	case en_stmt_prepared:
	  return SQL_NO_DATA_FOUND;
#endif

	case en_stmt_needdata:
	case en_stmt_mustput:
	case en_stmt_canput:
	  PUSHSQLERR (pstmt->herr, en_S1010);
	  return SQL_ERROR;

	default:
	  break;
	}
    }
  else if (pstmt->asyn_on != en_MoreResults)
    {
      PUSHSQLERR (pstmt->herr, en_S1010);

      return SQL_ERROR;
    }

  /* call driver */
  hproc = _iodbcdm_getproc (pstmt->hdbc, en_MoreResults);

  if (hproc == SQL_NULL_HPROC)
    {
      PUSHSQLERR (pstmt->herr, en_IM001);

      return SQL_ERROR;
    }

  CALL_DRIVER (pstmt->hdbc, pstmt, retcode, hproc,
      (pstmt->dhstmt));

  /* state transition */
  if (pstmt->asyn_on == en_MoreResults)
    {
      switch (retcode)
	{
	case SQL_SUCCESS:
	case SQL_SUCCESS_WITH_INFO:
	case SQL_NO_DATA_FOUND:
	case SQL_ERROR:
	  pstmt->asyn_on = en_NullProc;
	  break;

	case SQL_STILL_EXECUTING:
	default:
	  return retcode;
	}
    }

  switch (pstmt->state)
    {
    case en_stmt_allocated:
    case en_stmt_prepared:
      /* driver should return SQL_NO_DATA_FOUND */
	  if (pstmt->prep_state)
	    {
	      pstmt->state = en_stmt_cursoropen;
	    }
	  else
	    {
	      pstmt->state = en_stmt_prepared;
	    }
      break;

    case en_stmt_executed_with_info:
    	_iodbcdm_do_cursoropen (pstmt);
	/* FALL THROUGH */

    case en_stmt_executed:
      if (retcode == SQL_NO_DATA_FOUND)
	{
	  if (pstmt->prep_state)
	    {
	      pstmt->state = en_stmt_prepared;
	    }
	  else
	    {
	      pstmt->state = en_stmt_cursoropen;
	    }
	}
      else if (retcode == SQL_STILL_EXECUTING)
	{
	  pstmt->asyn_on = en_MoreResults;
	}
      break;

    case en_stmt_cursoropen:
    case en_stmt_fetched:
    case en_stmt_xfetched:
      if (retcode == SQL_SUCCESS)
	{
	  break;
	}
      else if (retcode == SQL_NO_DATA_FOUND)
	{
	  if (pstmt->prep_state)
	    {
	      pstmt->state = en_stmt_prepared;
	    }
	  else
	    {
	      pstmt->state = en_stmt_allocated;
	    }
	}
      else if (retcode == SQL_STILL_EXECUTING)
	{
	  pstmt->asyn_on = en_MoreResults;
	}
      break;

    default:
      break;
    }

  return retcode;
}


SQLRETURN SQL_API
SQLMoreResults (SQLHSTMT hstmt)
{
  ENTER_STMT (hstmt,
    trace_SQLMoreResults (TRACE_ENTER, hstmt));

  retcode = SQLMoreResults_Internal (hstmt);

  LEAVE_STMT (hstmt,
    trace_SQLMoreResults (TRACE_LEAVE, hstmt));
}


SQLRETURN SQL_API
_iodbcdm_SetPos (
  SQLHSTMT		  hstmt,
  SQLSETPOSIROW		  irow, 
  SQLUSMALLINT		  fOption, 
  SQLUSMALLINT		  fLock)
{
  STMT (pstmt, hstmt);
  HPROC hproc;
  SQLRETURN retcode;
  sqlstcode_t sqlstat = en_00000;

  /* check argument value */
  if (fOption > SQL_ADD || fLock > SQL_LOCK_UNLOCK)
    {
      PUSHSQLERR (pstmt->herr, en_S1009);
      return SQL_ERROR;
    }

  /* check state */
  if (pstmt->asyn_on == en_NullProc)
    {
      switch (pstmt->state)
	{
	case en_stmt_allocated:
	case en_stmt_prepared:
	case en_stmt_needdata:
	case en_stmt_mustput:
	case en_stmt_canput:
	  sqlstat = en_S1010;
	  break;

	case en_stmt_executed_with_info:
	case en_stmt_executed:
	case en_stmt_cursoropen:
	  sqlstat = en_24000;
	  break;

	default:
	  break;
	}
    }
  else if (pstmt->asyn_on != en_SetPos)
    {
      sqlstat = en_S1010;
    }

  if (sqlstat != en_00000)
    {
      PUSHSQLERR (pstmt->herr, sqlstat);

      return SQL_ERROR;
    }

  /* call driver */
  hproc = _iodbcdm_getproc (pstmt->hdbc, en_SetPos);

  if (hproc == SQL_NULL_HPROC)
    {
      PUSHSQLERR (pstmt->herr, en_IM001);

      return SQL_ERROR;
    }

  CALL_DRIVER (pstmt->hdbc, pstmt, retcode, hproc,
      (pstmt->dhstmt, irow, fOption, fLock));

  /* state transition */
  if (pstmt->asyn_on == en_SetPos)
    {
      switch (retcode)
	{
	case SQL_SUCCESS:
	case SQL_SUCCESS_WITH_INFO:
	case SQL_NEED_DATA:
	case SQL_ERROR:
	  pstmt->asyn_on = en_NullProc;
	  break;

	case SQL_STILL_EXECUTING:
	default:
	  return retcode;
	}
    }

  /* now, the only possible init state is 'xfetched' */
  switch (retcode)
    {
    case SQL_SUCCESS:
    case SQL_SUCCESS_WITH_INFO:
      break;

    case SQL_NEED_DATA:
      pstmt->state = en_stmt_needdata;
      pstmt->need_on = en_SetPos;
      break;

    case SQL_STILL_EXECUTING:
      pstmt->asyn_on = en_SetPos;
      break;

    default:
      break;
    }

  return retcode;
}


SQLRETURN SQL_API
SQLSetPos (
  SQLHSTMT		  hstmt,
  SQLSETPOSIROW		  irow, 
  SQLUSMALLINT		  fOption, 
  SQLUSMALLINT		  fLock)
{
  ENTER_STMT (hstmt,
    trace_SQLSetPos (TRACE_ENTER,
      hstmt, irow, fOption, fLock));

  retcode = _iodbcdm_SetPos (hstmt, irow, fOption, fLock);

  LEAVE_STMT (hstmt,
    trace_SQLSetPos (TRACE_LEAVE,
      hstmt, irow, fOption, fLock));
}
