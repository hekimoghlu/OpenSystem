/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
#include "gui.h"


void SQL_API
_iodbcdm_nativeerrorbox (
    HWND	hwnd,
    HENV	henv,
    HDBC	hdbc,
    HSTMT	hstmt)
{
  SQLCHAR buf[250];
  SQLCHAR sqlstate[15];

  /*
   * Get statement errors
   */
  if (SQLError (henv, hdbc, hstmt, sqlstate, NULL,
	  buf, sizeof (buf), NULL) == SQL_SUCCESS)
    create_error (hwnd, "Native ODBC Error", (LPCSTR) sqlstate, (LPCSTR) buf);

  /*
   * Get connection errors
   */
  if (SQLError (henv, hdbc, SQL_NULL_HSTMT, sqlstate,
	  NULL, buf, sizeof (buf), NULL) == SQL_SUCCESS)
    create_error (hwnd, "Native ODBC Error", (LPCSTR) sqlstate, (LPCSTR) buf);

  /*
   * Get environmental errors
   */
  if (SQLError (henv, SQL_NULL_HDBC, SQL_NULL_HSTMT,
	  sqlstate, NULL, buf, sizeof (buf), NULL) == SQL_SUCCESS)
    create_error (hwnd, "Native ODBC Error", (LPCSTR) sqlstate, (LPCSTR) buf);
}


void SQL_API
_iodbcdm_errorbox (
    HWND	hwnd,
    LPCSTR	szDSN,
    LPCSTR	szText)
{
  char msg[4096];

  if (SQLInstallerError (1, NULL, msg, sizeof (msg), NULL) == SQL_SUCCESS)
    create_error (hwnd, szDSN, szText, msg);
}


void SQL_API
_iodbcdm_errorboxw (
    HWND hwnd,
    LPCWSTR szDSN,
    LPCWSTR szText)
{
  wchar_t msg[4096];

  if (SQLInstallerErrorW (1, NULL, msg, sizeof (msg) / sizeof(wchar_t), NULL) == SQL_SUCCESS)
    create_errorw (hwnd, szDSN, szText, msg);
}


void SQL_API
_iodbcdm_messagebox (
    HWND	hwnd,
    LPCSTR	szDSN,
    LPCSTR	szText)
{
  create_message (hwnd, szDSN, szText);
}


void SQL_API
_iodbcdm_messageboxw (
    HWND hwnd,
    LPCWSTR szDSN,
    LPCWSTR szText)
{
  create_messagew (hwnd, szDSN, szText);
}


BOOL SQL_API
_iodbcdm_confirmbox (
    HWND	hwnd,
    LPCSTR	szDSN,
    LPCSTR	szText)
{
  return create_confirm (hwnd, (SQLPOINTER) szDSN, (SQLPOINTER) szText);
}


BOOL SQL_API
_iodbcdm_confirmboxw (
    HWND hwnd,
	 LPCWSTR szDSN,
	 LPCWSTR szText)
{
  return create_confirmw (hwnd, (SQLPOINTER)szDSN, (SQLPOINTER)szText);
}

