/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
#include "db_config.h"

#include "db_int.h"
#include "dbinc/db_page.h"
#include "dbinc/db_am.h"
#include "dbinc/mp.h"

static int __env_lsn_reset __P((ENV *, DB_THREAD_INFO *, const char *, int));

/*
 * __env_lsn_reset_pp --
 *	ENV->lsn_reset pre/post processing.
 *
 * PUBLIC: int __env_lsn_reset_pp __P((DB_ENV *, const char *, u_int32_t));
 */
int
__env_lsn_reset_pp(dbenv, name, flags)
	DB_ENV *dbenv;
	const char *name;
	u_int32_t flags;
{
	DB_THREAD_INFO *ip;
	ENV *env;
	int ret;

	env = dbenv->env;

	ENV_ILLEGAL_BEFORE_OPEN(env, "DB_ENV->lsn_reset");

	/*
	 * !!!
	 * The actual argument checking is simple, do it inline, outside of
	 * the replication block.
	 */
	if (flags != 0 && flags != DB_ENCRYPT)
		return (__db_ferr(env, "DB_ENV->lsn_reset", 0));

	ENV_ENTER(env, ip);
	REPLICATION_WRAP(env,
	    (__env_lsn_reset(env, ip, name, LF_ISSET(DB_ENCRYPT) ? 1 : 0)),
	    1, ret);
	ENV_LEAVE(env, ip);
	return (ret);
}

/*
 * __env_lsn_reset --
 *	Reset the LSNs for every page in the file.
 */
static int
__env_lsn_reset(env, ip, name, encrypted)
	ENV *env;
	DB_THREAD_INFO *ip;
	const char *name;
	int encrypted;
{
	DB *dbp;
	DB_MPOOLFILE *mpf;
	PAGE *pagep;
	db_pgno_t pgno;
	int t_ret, ret;

	/* Create the DB object. */
	if ((ret = __db_create_internal(&dbp, env, 0)) != 0)
		return (ret);

	/* If configured with a password, the databases are encrypted. */
	if (encrypted && (ret = __db_set_flags(dbp, DB_ENCRYPT)) != 0)
		goto err;

	/*
	 * Open the DB file.
	 *
	 * !!!
	 * Note DB_RDWRMASTER flag, we need to open the master database file
	 * for writing in this case.
	 */
	if ((ret = __db_open(dbp, ip, NULL,
	    name, NULL, DB_UNKNOWN, DB_RDWRMASTER, 0, PGNO_BASE_MD)) != 0) {
		__db_err(env, ret, "%s", name);
		goto err;
	}

	/* Reset the LSN on every page of the database file. */
	mpf = dbp->mpf;
	for (pgno = 0;
	    (ret = __memp_fget(mpf,
	    &pgno, ip, NULL, DB_MPOOL_DIRTY, &pagep)) == 0;
	    ++pgno) {
		LSN_NOT_LOGGED(pagep->lsn);
		if ((ret = __memp_fput(mpf,
		    ip, pagep, DB_PRIORITY_UNCHANGED)) != 0)
			goto err;
	}

	if (ret == DB_PAGE_NOTFOUND)
		ret = 0;

err:	if ((t_ret = __db_close(dbp, NULL, 0)) != 0 && ret == 0)
		ret = t_ret;
	return (ret);
}
