/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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

/*
 * __os_unique_id --
 *	Return a unique 32-bit value.
 *
 * PUBLIC: void __os_unique_id __P((ENV *, u_int32_t *));
 */
void
__os_unique_id(env, idp)
	ENV *env;
	u_int32_t *idp;
{
	DB_ENV *dbenv;
	db_timespec v;
	pid_t pid;
	u_int32_t id;

	*idp = 0;

	dbenv = env == NULL ? NULL : env->dbenv;

	/*
	 * Our randomized value is comprised of our process ID, the current
	 * time of day and a stack address, all XOR'd together.
	 */
	__os_id(dbenv, &pid, NULL);
	__os_gettime(env, &v, 1);

	id = (u_int32_t)pid ^
	    (u_int32_t)v.tv_sec ^ (u_int32_t)v.tv_nsec ^ P_TO_UINT32(&pid);

	/*
	 * We could try and find a reasonable random-number generator, but
	 * that's not all that easy to do.  Seed and use srand()/rand(), if
	 * we can find them.
	 */
	if (DB_GLOBAL(uid_init) == 0) {
		DB_GLOBAL(uid_init) = 1;
		srand((u_int)id);
	}
	id ^= (u_int)rand();

	*idp = id;
}
