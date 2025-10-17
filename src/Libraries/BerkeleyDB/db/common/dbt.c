/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
 * __dbt_usercopy --
 *	Take a copy of the user's data, if a callback is supplied.
 *
 * PUBLIC: int __dbt_usercopy __P((ENV *, DBT *));
 */
int
__dbt_usercopy(env, dbt)
	ENV *env;
	DBT *dbt;
{
	void *buf;
	int ret;

	if (dbt == NULL || !F_ISSET(dbt, DB_DBT_USERCOPY) || dbt->size == 0 ||
	    dbt->data != NULL)
		return (0);

	buf = NULL;
	if ((ret = __os_umalloc(env, dbt->size, &buf)) != 0 ||
	    (ret = env->dbt_usercopy(dbt, 0, buf, dbt->size,
	    DB_USERCOPY_GETDATA)) != 0)
		goto err;
	dbt->data = buf;

	return (0);

err:	if (buf != NULL) {
		__os_ufree(env, buf);
		dbt->data = NULL;
	}

	return (ret);
}

/*
 * __dbt_userfree --
 *	Free a copy of the user's data, if necessary.
 *
 * PUBLIC: void __dbt_userfree __P((ENV *, DBT *, DBT *, DBT *));
 */
void
__dbt_userfree(env, key, pkey, data)
	ENV *env;
	DBT *key, *pkey, *data;
{
	if (key != NULL &&
	    F_ISSET(key, DB_DBT_USERCOPY) && key->data != NULL) {
		__os_ufree(env, key->data);
		key->data = NULL;
	}
	if (pkey != NULL &&
	    F_ISSET(pkey, DB_DBT_USERCOPY) && pkey->data != NULL) {
		__os_ufree(env, pkey->data);
		pkey->data = NULL;
	}
	if (data != NULL &&
	    F_ISSET(data, DB_DBT_USERCOPY) && data->data != NULL) {
		__os_ufree(env, data->data);
		data->data = NULL;
	}
}
