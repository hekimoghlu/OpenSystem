/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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
/*
 * sqlutils.h - some wrapper for sql3lite
 */
#ifndef _SECURITY_UTILITIES_SQLUTILS_H_
#define _SECURITY_UTILITIES_SQLUTILS_H_

#include <sqlite3.h>

/* Those are just wrapper around the sqlite3 functions, but they have size_t for some len parameters,
   and checks for overflow before casting to int */
static inline int sqlite3_bind_blob_wrapper(sqlite3_stmt* pStmt, int i, const void* zData, size_t n, void(*xDel)(void*))
{
    if(n>INT_MAX) return SQLITE_TOOBIG;
    return sqlite3_bind_blob(pStmt, i, zData, (int)n, xDel);
}

static inline int sqlite3_bind_text_wrapper(sqlite3_stmt* pStmt, int i, const void* zData, size_t n, void(*xDel)(void*))
{
    if(n>INT_MAX) return SQLITE_TOOBIG;
    return sqlite3_bind_text(pStmt, i, zData, (int)n, xDel);
}

static inline int sqlite3_prepare_wrapper(sqlite3 *db, const char *zSql, size_t nByte, sqlite3_stmt **ppStmt, const char **pzTail)
{
    if(nByte>INT_MAX) return SQLITE_TOOBIG;
    return sqlite3_prepare(db, zSql, (int)nByte, ppStmt, pzTail);
}

#endif
