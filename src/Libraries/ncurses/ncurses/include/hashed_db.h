/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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
/****************************************************************************
 *  Author: Thomas E. Dickey                        2006                    *
 ****************************************************************************/

/*
 * $Id: hashed_db.h,v 1.6 2014/04/12 22:41:10 tom Exp $
 */

#ifndef HASHED_DB_H
#define HASHED_DB_H 1

#include <curses.h>

#if USE_HASHED_DB

#define DB_DBM_HSEARCH 0	/* quiet gcc -Wundef with db6 */

#include <db.h>

#ifndef DBN_SUFFIX
#define DBM_SUFFIX ".db"
#endif

#ifdef DB_VERSION_MAJOR
#define HASHED_DB_API DB_VERSION_MAJOR
#else
#define HASHED_DB_API 1		/* e.g., db 1.8.5 */
#endif

extern NCURSES_EXPORT(DB *) _nc_db_open(const char * /* path */, bool /* modify */);
extern NCURSES_EXPORT(bool) _nc_db_have_data(DBT * /* key */, DBT * /* data */, char ** /* buffer */, int * /* size */);
extern NCURSES_EXPORT(bool) _nc_db_have_index(DBT * /* key */, DBT * /* data */, char ** /* buffer */, int * /* size */);
extern NCURSES_EXPORT(int) _nc_db_close(DB * /* db */);
extern NCURSES_EXPORT(int) _nc_db_first(DB * /* db */, DBT * /* key */, DBT * /* data */);
extern NCURSES_EXPORT(int) _nc_db_next(DB * /* db */, DBT * /* key */, DBT * /* data */);
extern NCURSES_EXPORT(int) _nc_db_get(DB * /* db */, DBT * /* key */, DBT * /* data */);
extern NCURSES_EXPORT(int) _nc_db_put(DB * /* db */, DBT * /* key */, DBT * /* data */);

#endif

#endif /* HASHED_DB_H */
