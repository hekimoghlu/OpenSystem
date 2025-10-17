/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
#include "iodbcadm.h"

#if defined(__BEOS__)
#  include "be/gui.h"
#elif defined(_MAC)
#  include "mac/gui.h"
#elif defined(__GTK__)
#  include "gtk/gui.h"
#elif defined(_MACX)
#  include "macosx/gui.h"
#else
#  error GUI for this platform not supported ...
#endif

#ifndef	_GUI_H
#define _GUI_H

BOOL create_confirm (HWND hwnd, LPCSTR dsn, LPCSTR text);
BOOL create_confirmw (HWND hwnd, LPCWSTR dsn, LPCWSTR text);

#if 0
 void create_login (HWND hwnd, LPCSTR username, LPCSTR password, LPCSTR dsn,
     TLOGIN * log_t);
#endif

void create_dsnchooser (HWND hwnd, TDSNCHOOSER * choose_t);
void create_driverchooser (HWND hwnd, TDRIVERCHOOSER * choose_t);
void create_fdriverchooser (HWND hwnd, TFDRIVERCHOOSER * choose_t);
void create_translatorchooser (HWND hwnd, TTRANSLATORCHOOSER * choose_t);
void create_administrator (HWND hwnd);
void create_error (HWND hwnd, LPCSTR dsn, LPCSTR text, LPCSTR errmsg);
void create_errorw (HWND hwnd, LPCWSTR dsn, LPCWSTR text, LPCWSTR errmsg);
void create_message (HWND hwnd, LPCSTR dsn, LPCSTR text);
void create_messagew (HWND hwnd, LPCWSTR dsn, LPCWSTR text);
LPSTR create_driversetup (HWND hwnd, LPCSTR driver, LPCSTR attrs, BOOL add, BOOL user);
LPSTR create_filedsn (HWND hwnd);
BOOL create_connectionpool (HWND hwnd, TCONNECTIONPOOLING *choose_t);

typedef SQLRETURN SQL_API (*pSQLGetInfoFunc) (SQLHDBC hdbc,
    SQLUSMALLINT fInfoType, SQLPOINTER rgbInfoValue,
    SQLSMALLINT cbInfoValueMax, SQLSMALLINT * pcbInfoValue);
typedef SQLRETURN SQL_API (*pSQLAllocHandle) (SQLSMALLINT hdl_type,
    SQLHANDLE hdl_in, SQLHANDLE * hdl_out);
typedef SQLRETURN SQL_API (*pSQLAllocEnv) (SQLHENV * henv);
typedef SQLRETURN SQL_API (*pSQLAllocConnect) (SQLHENV henv, SQLHDBC * hdbc);
typedef SQLRETURN SQL_API (*pSQLFreeHandle) (SQLSMALLINT hdl_type,
    SQLHANDLE hdl_in);
typedef SQLRETURN SQL_API (*pSQLFreeEnv) (SQLHENV henv);
typedef SQLRETURN SQL_API (*pSQLFreeConnect) (SQLHDBC hdbc);
	
#endif
