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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)ndbm.c	8.4 (Berkeley) 7/21/94";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/db/hash/ndbm.c,v 1.7 2007/01/09 00:27:51 imp Exp $");

/*
 * This package provides a dbm compatible interface to the new hashing
 * package described in db(3).
 */

#include <sys/param.h>

#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <db.h>
#define _DBM
typedef DB DBM;
#include <ndbm.h>
#include "hash.h"

/*
 * Returns:
 * 	*DBM on success
 *	 NULL on failure
 */
extern DBM *
dbm_open(const char *file, int flags, mode_t mode)
{
	HASHINFO info;
	char path[MAXPATHLEN];

	info.bsize = 4096;
	info.ffactor = 40;
	info.nelem = 1;
	info.cachesize = 0;
	info.hash = NULL;
	info.lorder = 0;

	if( strlen(file) >= sizeof(path) - strlen(DBM_SUFFIX)) {
		errno = ENAMETOOLONG;
		return(NULL);
	}
	(void)strcpy(path, file);
	(void)strcat(path, DBM_SUFFIX);
	return ((DBM *)__hash_open(path, flags, mode, &info, 0));
}

extern void
dbm_close(DBM *db)
{
	(void)(db->close)(db);
}

/*
 * Returns:
 *	DATUM on success
 *	NULL on failure
 */
extern datum
dbm_fetch(DBM *db, datum key)
{
	datum retdata;
	int status;
	DBT dbtkey, dbtretdata;

	dbtkey.data = key.dptr;
	dbtkey.size = key.dsize;
	status = (db->get)(db, &dbtkey, &dbtretdata, 0);
	if (status) {
		dbtretdata.data = NULL;
		dbtretdata.size = 0;
	}
	retdata.dptr = dbtretdata.data;
	retdata.dsize = dbtretdata.size;
	return (retdata);
}

/*
 * Returns:
 *	DATUM on success
 *	NULL on failure
 */
extern datum
dbm_firstkey(DBM *db)
{
	int status;
	datum retkey;
	DBT dbtretkey, dbtretdata;
	HTAB *htab = (HTAB *)(db->internal);

	status = (db->seq)(db, &dbtretkey, &dbtretdata, R_FIRST);
	if (status) {
		dbtretkey.data = NULL;
		htab->nextkey_eof = 1;
	} else
		htab->nextkey_eof = 0;
	retkey.dptr = dbtretkey.data;
	retkey.dsize = dbtretkey.size;
	return (retkey);
}

/*
 * Returns:
 *	DATUM on success
 *	NULL on failure
 */
extern datum
dbm_nextkey(DBM *db)
{
	int status = 1;
	datum retkey;
	DBT dbtretkey, dbtretdata;
	HTAB *htab = (HTAB *)(db->internal);

	if (htab->nextkey_eof)
		dbtretkey.data = NULL;
	else {
		status = (db->seq)(db, &dbtretkey, &dbtretdata, R_NEXT);
		if (status) {
			dbtretkey.data = NULL;
			htab->nextkey_eof = 1;
		}
	}
	retkey.dptr = dbtretkey.data;
	retkey.dsize = dbtretkey.size;
	return (retkey);
}

/*
 * Returns:
 *	 0 on success
 *	<0 failure
 */
extern int
dbm_delete(DBM *db, datum key)
{
	int status;
	DBT dbtkey;

	dbtkey.data = key.dptr;
	dbtkey.size = key.dsize;
	status = (db->del)(db, &dbtkey, 0);
	if (status)
		return (-1);
	else
		return (0);
}

/*
 * Returns:
 *	 0 on success
 *	<0 failure
 *	 1 if DBM_INSERT and entry exists
 */
extern int
dbm_store(DBM *db, datum key, datum data, int flags)
{
	DBT dbtkey, dbtdata;

	dbtkey.data = key.dptr;
	dbtkey.size = key.dsize;
	dbtdata.data = data.dptr;
	dbtdata.size = data.dsize;
	return ((db->put)(db, &dbtkey, &dbtdata,
	    (flags == DBM_INSERT) ? R_NOOVERWRITE : 0));
}

extern int
dbm_error(DBM *db)
{
	HTAB *hp;

	hp = (HTAB *)db->internal;
	return (hp->error);
}

extern int
dbm_clearerr(DBM *db)
{
	HTAB *hp;

	hp = (HTAB *)db->internal;
	hp->error = 0;
	return (0);
}

extern int
dbm_dirfno(DBM *db)
{
	return(((HTAB *)db->internal)->fp);
}
