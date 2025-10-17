/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#include <odbcinst.h>
#include <sql.h>
#include <sqltypes.h>
#include <sqlucode.h>

#ifndef	_IODBCADM_H
#define	_IODBCADM_H

#define USER_DSN		0
#define SYSTEM_DSN		1
#define FILE_DSN		2

SQLRETURN SQL_API iodbcdm_drvconn_dialbox (HWND hwnd, LPSTR szInOutConnStr,
    DWORD cbInOutConnStr, int *sqlStat, SQLUSMALLINT fDriverCompletion,
    UWORD * config);
SQLRETURN SQL_API iodbcdm_drvconn_dialboxw (HWND hwnd, LPWSTR szInOutConnStr,
    DWORD cbInOutConnStr, int *sqlStat, SQLUSMALLINT fDriverCompletion,
    UWORD * config);

SQLRETURN SQL_API _iodbcdm_drvchoose_dialbox (HWND hwnd, LPSTR szInOutDrvStr,
    DWORD cbInOutDrvStr, int *sqlStat);
SQLRETURN SQL_API _iodbcdm_drvchoose_dialboxw (HWND hwnd, LPWSTR szInOutConnStr,
    DWORD cbInOutConnStr, int * sqlStat);

SQLRETURN SQL_API _iodbcdm_trschoose_dialbox (HWND hwnd, LPSTR szInOutDrvStr,
    DWORD cbInOutDrvStr, int *sqlStat);
SQLRETURN SQL_API _iodbcdm_trschoose_dialboxw (HWND hwnd, LPWSTR szInOutDrvStr,
    DWORD cbInOutDrvStr, int * sqlStat);

void SQL_API _iodbcdm_errorbox (HWND hwnd, LPCSTR szDSN, LPCSTR szText);
void SQL_API _iodbcdm_errorboxw (HWND hwnd, LPCWSTR szDSN, LPCWSTR szText);
void SQL_API _iodbcdm_messagebox (HWND hwnd, LPCSTR szDSN, LPCSTR szText);
void SQL_API _iodbcdm_messageboxw (HWND hwnd, LPCWSTR szDSN, LPCWSTR szText);
BOOL SQL_API _iodbcdm_confirmbox (HWND hwnd, LPCSTR szDSN, LPCSTR szText);
BOOL SQL_API _iodbcdm_confirmboxw (HWND hwnd, LPCWSTR szDSN, LPCWSTR szText);
void _iodbcdm_nativeerrorbox (HWND hwnd, HENV henv, HDBC hdbc, HSTMT hstmt);

SQLRETURN SQL_API _iodbcdm_admin_dialbox (HWND hwnd);

typedef SQLRETURN SQL_API (*pAdminBoxFunc) (HWND hwnd);
typedef SQLRETURN SQL_API (*pTrsChooseFunc) (HWND hwnd, LPSTR szInOutDrvStr,
    DWORD cbInOutDrvStr, int *sqlStat);
typedef SQLRETURN SQL_API (*pDrvConnFunc) (HWND hwnd, LPSTR szInOutConnStr,
    DWORD cbInOutConnStr, int *sqlStat, SQLUSMALLINT fDriverCompletion,
    UWORD * config);
typedef SQLRETURN SQL_API (*pDrvConnWFunc) (HWND hwnd, LPWSTR szInOutConnStr,
    DWORD cbInOutConnStr, int * sqlStat, SQLUSMALLINT fDriverCompletion, 
    UWORD *config);
#endif
