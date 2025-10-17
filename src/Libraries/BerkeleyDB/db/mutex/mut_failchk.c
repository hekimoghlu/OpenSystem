/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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
#include "dbinc/mutex_int.h"

/*
 * __mut_failchk --
 *	Check for mutexes held by dead processes.
 *
 * PUBLIC: int __mut_failchk __P((ENV *));
 */
int
__mut_failchk(env)
	ENV *env;
{
	DB_ENV *dbenv;
	DB_MUTEX *mutexp;
	DB_MUTEXMGR *mtxmgr;
	DB_MUTEXREGION *mtxregion;
	db_mutex_t i;
	int ret;
	char buf[DB_THREADID_STRLEN];

	dbenv = env->dbenv;
	mtxmgr = env->mutex_handle;
	mtxregion = mtxmgr->reginfo.primary;
	ret = 0;

	MUTEX_SYSTEM_LOCK(env);
	for (i = 1; i <= mtxregion->stat.st_mutex_cnt; ++i, ++mutexp) {
		mutexp = MUTEXP_SET(i);

		/*
		 * We're looking for per-process mutexes where the process
		 * has died.
		 */
		if (!F_ISSET(mutexp, DB_MUTEX_ALLOCATED) ||
		    !F_ISSET(mutexp, DB_MUTEX_PROCESS_ONLY))
			continue;

		/*
		 * The thread that allocated the mutex may have exited, but
		 * we cannot reclaim the mutex if the process is still alive.
		 */
		if (dbenv->is_alive(
		    dbenv, mutexp->pid, 0, DB_MUTEX_PROCESS_ONLY))
			continue;

		__db_msg(env, "Freeing mutex for process: %s",
		    dbenv->thread_id_string(dbenv, mutexp->pid, 0, buf));

		/* Unlock and free the mutex. */
		if (F_ISSET(mutexp, DB_MUTEX_LOCKED))
			MUTEX_UNLOCK(env, i);

		if ((ret = __mutex_free_int(env, 0, &i)) != 0)
			break;
	}
	MUTEX_SYSTEM_UNLOCK(env);

	return (ret);
}
