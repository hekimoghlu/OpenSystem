/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
/* $Id$ */

#ifndef __ndbm_wrap_h__
#define __ndbm_wrap_h__

#include <stdio.h>
#include <sys/types.h>

#ifndef ROKEN_LIB_FUNCTION
#ifdef _WIN32
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL     __cdecl
#else
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL
#endif
#endif

#ifndef dbm_rename
#define dbm_rename(X)	__roken_ ## X
#endif

#define dbm_open	dbm_rename(dbm_open)
#define dbm_close	dbm_rename(dbm_close)
#define dbm_delete	dbm_rename(dbm_delete)
#define dbm_fetch	dbm_rename(dbm_fetch)
#define dbm_get		dbm_rename(dbm_get)
#define dbm_firstkey	dbm_rename(dbm_firstkey)
#define dbm_nextkey	dbm_rename(dbm_nextkey)
#define dbm_store	dbm_rename(dbm_store)
#define dbm_error	dbm_rename(dbm_error)
#define dbm_clearerr	dbm_rename(dbm_clearerr)

#define datum		dbm_rename(datum)

typedef struct {
    void *dptr;
    size_t dsize;
} datum;

#define DBM_REPLACE 1
typedef struct DBM DBM;

#if 0
typedef struct {
    int dummy;
} DBM;
#endif

ROKEN_LIB_FUNCTION int   ROKEN_LIB_CALL dbm_clearerr (DBM*);
ROKEN_LIB_FUNCTION void  ROKEN_LIB_CALL dbm_close (DBM*);
ROKEN_LIB_FUNCTION int   ROKEN_LIB_CALL dbm_delete (DBM*, datum);
ROKEN_LIB_FUNCTION int   ROKEN_LIB_CALL dbm_error (DBM*);
ROKEN_LIB_FUNCTION datum ROKEN_LIB_CALL dbm_fetch (DBM*, datum);
ROKEN_LIB_FUNCTION datum ROKEN_LIB_CALL dbm_firstkey (DBM*);
ROKEN_LIB_FUNCTION datum ROKEN_LIB_CALL dbm_nextkey (DBM*);
ROKEN_LIB_FUNCTION DBM*  ROKEN_LIB_CALL dbm_open (const char*, int, mode_t);
ROKEN_LIB_FUNCTION int   ROKEN_LIB_CALL dbm_store (DBM*, datum, datum, int);

#endif /* __ndbm_wrap_h__ */
