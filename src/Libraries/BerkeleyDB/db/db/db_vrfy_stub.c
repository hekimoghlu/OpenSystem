/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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
#ifndef HAVE_VERIFY
#include "db_config.h"

#include "db_int.h"
#include "dbinc/db_page.h"
#include "dbinc/db_am.h"
#include "dbinc/db_verify.h"

/*
 * If the library wasn't compiled with the verification support, various
 * routines aren't available.  Stub them here, returning an appropriate
 * error.
 */

static int __db_novrfy __P((ENV *));

/*
 * __db_novrfy --
 *	Error when a Berkeley DB build doesn't include the access method.
 */
static int
__db_novrfy(env)
	ENV *env;
{
	__db_errx(env,
	    "library build did not include support for database verification");
	return (DB_OPNOTSUP);
}

int
__db_verify_pp(dbp, file, database, outfile, flags)
	DB *dbp;
	const char *file, *database;
	FILE *outfile;
	u_int32_t flags;
{
	int ret;

	COMPQUIET(file, NULL);
	COMPQUIET(database, NULL);
	COMPQUIET(outfile, NULL);
	COMPQUIET(flags, 0);

	ret = __db_novrfy(dbp->env);

	/* The verify method is a destructor. */
	(void)__db_close(dbp, NULL, 0);

	return (ret);
}

int
__db_verify_internal(dbp, name, subdb, handle, callback, flags)
	DB *dbp;
	const char *name, *subdb;
	void *handle;
	int (*callback) __P((void *, const void *));
	u_int32_t flags;
{
	COMPQUIET(dbp, NULL);
	COMPQUIET(name, NULL);
	COMPQUIET(subdb, NULL);
	COMPQUIET(handle, NULL);
	COMPQUIET(callback, NULL);
	COMPQUIET(flags, 0);
	return (0);
}

int
__db_vrfy_getpageinfo(vdp, pgno, pipp)
	VRFY_DBINFO *vdp;
	db_pgno_t pgno;
	VRFY_PAGEINFO **pipp;
{
	COMPQUIET(pgno, 0);
	COMPQUIET(pipp, NULL);
	return (__db_novrfy(vdp->pgdbp->env));
}

int
__db_vrfy_putpageinfo(env, vdp, pip)
	ENV *env;
	VRFY_DBINFO *vdp;
	VRFY_PAGEINFO *pip;
{
	COMPQUIET(vdp, NULL);
	COMPQUIET(pip, NULL);
	return (__db_novrfy(env));
}
#endif /* !HAVE_VERIFY */
