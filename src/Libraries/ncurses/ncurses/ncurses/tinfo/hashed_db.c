/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
 *  Author: Thomas E. Dickey                        2006-on                 *
 ****************************************************************************/

#include <curses.priv.h>
#include <tic.h>
#include <hashed_db.h>

#if USE_HASHED_DB

MODULE_ID("$Id: hashed_db.c,v 1.17 2013/12/15 00:33:01 tom Exp $")

#if HASHED_DB_API >= 2
static DBC *cursor;
#endif

typedef struct _myconn {
    struct _myconn *next;
    DB *db;
    char *path;
    bool modify;
} MYCONN;

static MYCONN *connections;

static void
cleanup(void)
{
    while (connections != 0) {
	_nc_db_close(connections->db);
    }
}

static DB *
find_connection(const char *path, bool modify)
{
    DB *result = 0;
    MYCONN *p;

    for (p = connections; p != 0; p = p->next) {
	if (!strcmp(p->path, path) && p->modify == modify) {
	    result = p->db;
	    break;
	}
    }

    return result;
}

static void
drop_connection(DB * db)
{
    MYCONN *p, *q;

    for (p = connections, q = 0; p != 0; q = p, p = p->next) {
	if (p->db == db) {
	    if (q != 0)
		q->next = p->next;
	    else
		connections = p->next;
	    free(p->path);
	    free(p);
	    break;
	}
    }
}

static void
make_connection(DB * db, const char *path, bool modify)
{
    MYCONN *p = typeCalloc(MYCONN, 1);

    if (p != 0) {
	p->db = db;
	p->path = strdup(path);
	p->modify = modify;
	if (p->path != 0) {
	    p->next = connections;
	    connections = p;
	} else {
	    free(p);
	}
    }
}

/*
 * Open the database.
 */
NCURSES_EXPORT(DB *)
_nc_db_open(const char *path, bool modify)
{
    DB *result = 0;
    int code;

    if (connections == 0)
	atexit(cleanup);

    if ((result = find_connection(path, modify)) == 0) {

#if HASHED_DB_API >= 4
	db_create(&result, NULL, 0);
	if ((code = result->open(result,
				 NULL,
				 path,
				 NULL,
				 DB_HASH,
				 modify ? DB_CREATE : DB_RDONLY,
				 0644)) != 0) {
	    result = 0;
	}
#elif HASHED_DB_API >= 3
	db_create(&result, NULL, 0);
	if ((code = result->open(result,
				 path,
				 NULL,
				 DB_HASH,
				 modify ? DB_CREATE : DB_RDONLY,
				 0644)) != 0) {
	    result = 0;
	}
#elif HASHED_DB_API >= 2
	if ((code = db_open(path,
			    DB_HASH,
			    modify ? DB_CREATE : DB_RDONLY,
			    0644,
			    (DB_ENV *) 0,
			    (DB_INFO *) 0,
			    &result)) != 0) {
	    result = 0;
	}
#else
	if ((result = dbopen(path,
			     modify ? (O_CREAT | O_RDWR) : O_RDONLY,
			     0644,
			     DB_HASH,
			     NULL)) == 0) {
	    code = errno;
	}
#endif
	if (result != 0) {
	    make_connection(result, path, modify);
	    T(("opened %s", path));
	} else {
	    T(("cannot open %s: %s", path, strerror(code)));
	}
    }
    return result;
}

/*
 * Close the database.  Do not attempt to use the 'db' handle after this call.
 */
NCURSES_EXPORT(int)
_nc_db_close(DB * db)
{
    int result;

    drop_connection(db);
#if HASHED_DB_API >= 2
    result = db->close(db, 0);
#else
    result = db->close(db);
#endif
    return result;
}

/*
 * Write a record to the database.
 *
 * Returns 0 on success.
 *
 * FIXME:  the FreeBSD cap_mkdb program assumes the database could have
 * duplicates.  There appears to be no good reason for that (review/fix).
 */
NCURSES_EXPORT(int)
_nc_db_put(DB * db, DBT * key, DBT * data)
{
    int result;
#if HASHED_DB_API >= 2
    /* remove any pre-existing value, since we do not want duplicates */
    (void) db->del(db, NULL, key, 0);
    result = db->put(db, NULL, key, data, DB_NOOVERWRITE);
#else
    result = db->put(db, key, data, R_NOOVERWRITE);
#endif
    return result;
}

/*
 * Read a record from the database.
 *
 * Returns 0 on success.
 */
NCURSES_EXPORT(int)
_nc_db_get(DB * db, DBT * key, DBT * data)
{
    int result;

    memset(data, 0, sizeof(*data));
#if HASHED_DB_API >= 2
    result = db->get(db, NULL, key, data, 0);
#else
    result = db->get(db, key, data, 0);
#endif
    return result;
}

/*
 * Read the first record from the database, ignoring order.
 *
 * Returns 0 on success.
 */
NCURSES_EXPORT(int)
_nc_db_first(DB * db, DBT * key, DBT * data)
{
    int result;

    memset(key, 0, sizeof(*key));
    memset(data, 0, sizeof(*data));
#if HASHED_DB_API >= 2
    if ((result = db->cursor(db, NULL, &cursor, 0)) == 0) {
	result = cursor->c_get(cursor, key, data, DB_FIRST);
    }
#else
    result = db->seq(db, key, data, 0);
#endif
    return result;
}

/*
 * Read the next record from the database, ignoring order.
 *
 * Returns 0 on success.
 */
NCURSES_EXPORT(int)
_nc_db_next(DB * db, DBT * key, DBT * data)
{
    int result;

#if HASHED_DB_API >= 2
    (void) db;
    if (cursor != 0) {
	result = cursor->c_get(cursor, key, data, DB_NEXT);
    } else {
	result = -1;
    }
#else
    result = db->seq(db, key, data, 0);
#endif
    return result;
}

/*
 * Check if a record is a terminfo index record.  Index records are those that
 * contain only an alias pointing to a list of aliases.
 */
NCURSES_EXPORT(bool)
_nc_db_have_index(DBT * key, DBT * data, char **buffer, int *size)
{
    bool result = FALSE;
    int used = (int) data->size - 1;
    char *have = (char *) data->data;

    (void) key;
    if (*have++ == 2) {
	result = TRUE;
    }
    /*
     * Update params in any case for consistency with _nc_db_have_data().
     */
    *buffer = have;
    *size = used;
    return result;
}

/*
 * Check if a record is the terminfo data record.  Ignore index records, e.g.,
 * those that contain only an alias pointing to a list of aliases.
 */
NCURSES_EXPORT(bool)
_nc_db_have_data(DBT * key, DBT * data, char **buffer, int *size)
{
    bool result = FALSE;
    int used = (int) data->size - 1;
    char *have = (char *) data->data;

    if (*have++ == 0) {
	if (data->size > key->size
	    && IS_TIC_MAGIC(have)) {
	    result = TRUE;
	}
    }
    /*
     * Update params in any case to make it simple to follow a index record
     * to the data record.
     */
    *buffer = have;
    *size = used;
    return result;
}

#else

extern
NCURSES_EXPORT(void)
_nc_hashed_db(void);

NCURSES_EXPORT(void)
_nc_hashed_db(void)
{
}

#endif /* USE_HASHED_DB */
