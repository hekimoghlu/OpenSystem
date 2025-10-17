/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include "dbinc/log.h"
#include "dbinc/mp.h"

/*
 * memp_register_pp --
 *	ENV->memp_register pre/post processing.
 *
 * PUBLIC: int __memp_register_pp __P((DB_ENV *, int,
 * PUBLIC:     int (*)(DB_ENV *, db_pgno_t, void *, DBT *),
 * PUBLIC:     int (*)(DB_ENV *, db_pgno_t, void *, DBT *)));
 */
int
__memp_register_pp(dbenv, ftype, pgin, pgout)
	DB_ENV *dbenv;
	int ftype;
	int (*pgin) __P((DB_ENV *, db_pgno_t, void *, DBT *));
	int (*pgout) __P((DB_ENV *, db_pgno_t, void *, DBT *));
{
	DB_THREAD_INFO *ip;
	ENV *env;
	int ret;

	env = dbenv->env;

	ENV_REQUIRES_CONFIG(env,
	    env->mp_handle, "DB_ENV->memp_register", DB_INIT_MPOOL);

	ENV_ENTER(env, ip);
	REPLICATION_WRAP(env,
	    (__memp_register(env, ftype, pgin, pgout)), 0, ret);
	ENV_LEAVE(env, ip);
	return (ret);
}

/*
 * memp_register --
 *	ENV->memp_register.
 *
 * PUBLIC: int __memp_register __P((ENV *, int,
 * PUBLIC:     int (*)(DB_ENV *, db_pgno_t, void *, DBT *),
 * PUBLIC:     int (*)(DB_ENV *, db_pgno_t, void *, DBT *)));
 */
int
__memp_register(env, ftype, pgin, pgout)
	ENV *env;
	int ftype;
	int (*pgin) __P((DB_ENV *, db_pgno_t, void *, DBT *));
	int (*pgout) __P((DB_ENV *, db_pgno_t, void *, DBT *));
{
	DB_MPOOL *dbmp;
	DB_MPREG *mpreg;
	int ret;

	dbmp = env->mp_handle;

	/*
	 * We keep the DB pgin/pgout functions outside of the linked list
	 * to avoid locking/unlocking the linked list on every page I/O.
	 *
	 * The Berkeley DB I/O conversion functions are registered when the
	 * environment is first created, so there's no need for locking here.
	 */
	if (ftype == DB_FTYPE_SET) {
		if (dbmp->pg_inout != NULL)
			return (0);
		if ((ret =
		    __os_malloc(env, sizeof(DB_MPREG), &dbmp->pg_inout)) != 0)
			return (ret);
		dbmp->pg_inout->ftype = ftype;
		dbmp->pg_inout->pgin = pgin;
		dbmp->pg_inout->pgout = pgout;
		return (0);
	}

	/*
	 * The item may already have been registered.  If already registered,
	 * just update the entry, although it's probably unchanged.
	 */
	MUTEX_LOCK(env, dbmp->mutex);
	LIST_FOREACH(mpreg, &dbmp->dbregq, q)
		if (mpreg->ftype == ftype) {
			mpreg->pgin = pgin;
			mpreg->pgout = pgout;
			break;
		}

	if (mpreg == NULL) {			/* New entry. */
		if ((ret = __os_malloc(env, sizeof(DB_MPREG), &mpreg)) != 0)
			return (ret);
		mpreg->ftype = ftype;
		mpreg->pgin = pgin;
		mpreg->pgout = pgout;

		LIST_INSERT_HEAD(&dbmp->dbregq, mpreg, q);
	}
	MUTEX_UNLOCK(env, dbmp->mutex);

	return (0);
}
