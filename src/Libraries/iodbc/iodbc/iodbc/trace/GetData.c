/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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
#include "trace.h"


void
_trace_data (
  SQLSMALLINT		  fCType,
  SQLPOINTER		  rgbValue,
  SQLLEN		  cbValueMax,
  SQLLEN	    	* pcbValue,
  int			  output)
{
  char buf[1024];		/* Temp buffer */

  if (!rgbValue)
    {
      trace_emit ("\t\t%-15.15s   0x0\n", "SQLPOINTER");
      return;
    }

  trace_emit ("\t\t%-15.15s   %p\n", "SQLPOINTER", rgbValue);

  if (!output)
    return;			/* Only print buffer content on leave */

  switch (fCType)
    {
    case SQL_C_BINARY:
		{
			int len=cbValueMax;
			if (pcbValue) {
				len = *((SQLINTEGER *) pcbValue);
				if (len>cbValueMax)
					len = cbValueMax;
			}
			trace_emit_binary ((unsigned char *) rgbValue, len);
		}
      break;

    case SQL_C_BIT:
      {
	int i = (int) *(char *) rgbValue;
	sprintf (buf, "%d", i > 0 ? 1 : 0);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_CHAR:
      {
		  int len=cbValueMax;
		  if (pcbValue) {
			  len = *((SQLINTEGER *) pcbValue);
			  if (len>cbValueMax)
				  len = cbValueMax;
		  }
		  trace_emit_string ((SQLCHAR *) rgbValue, len, 0);
      }
      break;

    case SQL_C_DATE:
#if ODBCVER >= 0x0300
    case SQL_C_TYPE_DATE:
#endif
      {
	DATE_STRUCT *d = (DATE_STRUCT *) rgbValue;
	sprintf (buf, "%04d-%02d-%02d", d->year, d->month, d->day);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_DEFAULT:
      /*
       *  Not enough information to dump the content of the buffer
       */
      return;

    case SQL_C_DOUBLE:
      {
	double d = *(double *) rgbValue;
	sprintf (buf, "%f", d);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_FLOAT:
      {
	float f = *(float *) rgbValue;
	sprintf (buf, "%f", f);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

#if (ODBCVER >= 0x0350)
    case SQL_C_GUID:
      {
	SQLGUID *g = (SQLGUID *) rgbValue;
	sprintf (buf,
	    "%08lX-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X",
	    (unsigned long) g->Data1,
	    g->Data2, g->Data3,
	    g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3],
            g->Data4[4], g->Data4[5], g->Data4[6], g->Data4[7]);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;
#endif

#if ODBCVER >= 0x0300
    case SQL_C_INTERVAL_DAY:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu days",
	    (unsigned long) i->intval.day_second.day);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_DAY_TO_HOUR:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu days %lu hours",
	    (unsigned long) i->intval.day_second.day,
	    (unsigned long) i->intval.day_second.hour);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_DAY_TO_MINUTE:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu days %lu hours %lu minutes",
	    (unsigned long) i->intval.day_second.day,
	    (unsigned long) i->intval.day_second.hour,
	    (unsigned long) i->intval.day_second.minute);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_DAY_TO_SECOND:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu days %lu hours %lu minutes %lu seconds",
	    (unsigned long) i->intval.day_second.day,
	    (unsigned long) i->intval.day_second.hour,
	    (unsigned long) i->intval.day_second.minute,
	    (unsigned long) i->intval.day_second.second);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_HOUR:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu hours",
	    (unsigned long) i->intval.day_second.hour);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_HOUR_TO_MINUTE:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu hours %lu minutes",
	    (unsigned long) i->intval.day_second.hour,
	    (unsigned long) i->intval.day_second.minute);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_HOUR_TO_SECOND:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu hours %lu minutes %lu seconds",
	    (unsigned long) i->intval.day_second.hour,
	    (unsigned long) i->intval.day_second.minute,
	    (unsigned long) i->intval.day_second.second);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_MINUTE:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu minutes",
	    (unsigned long) i->intval.day_second.minute);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_MINUTE_TO_SECOND:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu minutes %lu seconds",
	    (unsigned long) i->intval.day_second.minute,
	    (unsigned long) i->intval.day_second.second);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_MONTH:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu months",
	    (unsigned long) i->intval.year_month.month);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_SECOND:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu seconds",
	    (unsigned long) i->intval.day_second.second);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_YEAR:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu years",
	    (unsigned long) i->intval.year_month.year);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_INTERVAL_YEAR_TO_MONTH:
      {
	SQL_INTERVAL_STRUCT *i = (SQL_INTERVAL_STRUCT *) rgbValue;
	sprintf (buf, "%lu years %lu months",
	    (unsigned long) i->intval.year_month.year,
	    (unsigned long) i->intval.year_month.month);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;
#endif

    case SQL_C_LONG:
    case SQL_C_SLONG:
      {
	long l = *(long *) rgbValue;
	sprintf (buf, "%ld", l);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_ULONG:
      {
	unsigned long l = *(unsigned long *) rgbValue;
	sprintf (buf, "%lu", l);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;


#if ODBCVER >= 0x0300
    case SQL_C_NUMERIC:
      /* NOT YET */
      break;
#endif

#if ODBCVER >= 0x0300
    case SQL_C_SBIGINT:
#if defined (ODBCINT64)
      {
	ODBCINT64 l = *(ODBCINT64 *) rgbValue;
	sprintf (buf, "%lld", l);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
#endif
      break;

    case SQL_C_UBIGINT:
#if defined (ODBCINT64)
      {
	unsigned ODBCINT64 l = *(unsigned ODBCINT64 *) rgbValue;
	sprintf (buf, "%llu", l);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
#endif
      break;
#endif

    case SQL_C_SHORT:
    case SQL_C_SSHORT:
      {
	int i = (int) *(short *) rgbValue;
	sprintf (buf, "%d", i);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_USHORT:
      {
	unsigned int i = (unsigned int) *(unsigned short *) rgbValue;
	sprintf (buf, "%u", i);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_TIME:
#if ODBCVER >= 0x0300
    case SQL_C_TYPE_TIME:
#endif
      {
	TIME_STRUCT *t = (TIME_STRUCT *) rgbValue;
	sprintf (buf, "%02d:%02d:%02d", t->hour, t->minute, t->second);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_TIMESTAMP:
#if ODBCVER >= 0x0300
    case SQL_C_TYPE_TIMESTAMP:
#endif
      {
	TIMESTAMP_STRUCT *t = (TIMESTAMP_STRUCT *) rgbValue;
	sprintf (buf, "%04d-%02d-%02d %02d:%02d:%02d.%06ld",
	    t->year, t->month, t->day,
	    t->hour, t->minute, t->second, 
	    (long) t->fraction);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_TINYINT:
    case SQL_C_STINYINT:
      {
	int i = (int) *(char *) rgbValue;
	sprintf (buf, "%d", i);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_UTINYINT:
      {
	unsigned int i = (unsigned int) *(unsigned char *) rgbValue;
	sprintf (buf, "%u", i);
	trace_emit_string ((SQLCHAR *) buf, SQL_NTS, 0);
      }
      break;

    case SQL_C_WCHAR:
      {
	SQLCHAR *wstr;
        int len;
	if (pcbValue && cbValueMax > 0)
	  len = *((SQLINTEGER *) pcbValue);
	else
	  len = cbValueMax;
	wstr = dm_SQL_W2A ((wchar_t *) rgbValue, len);
	trace_emit_string (wstr, SQL_NTS, 1);
	free (wstr);
      }
      break;

    default:
      /*
       *  Unhandled/Unknown datatype
       */
      break;
    }

  return;
}


void
trace_SQLGetData (int trace_leave, int retcode,
  SQLHSTMT		  hstmt,
  SQLUSMALLINT		  icol,
  SQLSMALLINT		  fCType,
  SQLPOINTER		  rgbValue,
  SQLLEN		  cbValueMax,
  SQLLEN	    	* pcbValue)
{
  /* Trace function */
  _trace_print_function (en_GetData, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, hstmt);
  _trace_usmallint (icol);
  _trace_c_type (fCType);
  _trace_data (fCType, rgbValue, cbValueMax, pcbValue, TRACE_OUTPUT_SUCCESS);
  _trace_len (cbValueMax);
  _trace_len_p (pcbValue, TRACE_OUTPUT_SUCCESS);
}
