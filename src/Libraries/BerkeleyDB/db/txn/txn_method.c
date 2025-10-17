/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
#include "dbinc/txn.h"

/*
 * __txn_env_create --
 *	Transaction specific initialization of the DB_ENV structure.
 *
 * PUBLIC: int __txn_env_create __P((DB_ENV *));
 */
int
__txn_env_create(dbenv)
	DB_ENV *dbenv;
{
	/*
	 * !!!
	 * Our caller has not yet had the opportunity to reset the panic
	 * state or turn off mutex locking, and so we can neither check
	 * the panic state or acquire a mutex in the DB_ENV create path.
	 */
	dbenv->tx_max = DEF_MAX_TXNS;

	return (0);
}

/*
 * __txn_env_destroy --
 *	Transaction specific destruction of the DB_ENV structure.
 *
 * PUBLIC: void __txn_env_destroy __P((DB_ENV *));
 */
void
__txn_env_destroy(dbenv)
	DB_ENV *dbenv;
{
	COMPQUIET(dbenv, NULL);
}

/*
 * PUBLIC: int __txn_get_tx_max __P((DB_ENV *, u_int32_t *));
 */
int
__txn_get_tx_max(dbenv, tx_maxp)
	DB_ENV *dbenv;
	u_int32_t *tx_maxp;
{
	ENV *env;

	env = dbenv->env;

	ENV_NOT_CONFIGURED(env,
	    env->tx_handle, "DB_ENV->get_tx_max", DB_INIT_TXN);

	if (TXN_ON(env)) {
		/* Cannot be set after open, no lock required to read. */
		*tx_maxp = ((DB_TXNREGION *)
		    env->tx_handle->reginfo.primary)->maxtxns;
	} else
		*tx_maxp = dbenv->tx_max;
	return (0);
}

/*
 * __txn_set_tx_max --
 *	DB_ENV->set_tx_max.
 *
 * PUBLIC: int __txn_set_tx_max __P((DB_ENV *, u_int32_t));
 */
int
__txn_set_tx_max(dbenv, tx_max)
	DB_ENV *dbenv;
	u_int32_t tx_max;
{
	ENV *env;

	env = dbenv->env;

	ENV_ILLEGAL_AFTER_OPEN(env, "DB_ENV->set_tx_max");

	dbenv->tx_max = tx_max;
	return (0);
}

/*
 * PUBLIC: int __txn_get_tx_timestamp __P((DB_ENV *, time_t *));
 */
int
__txn_get_tx_timestamp(dbenv, timestamp)
	DB_ENV *dbenv;
	time_t *timestamp;
{
	*timestamp = dbenv->tx_timestamp;
	return (0);
}

/*
 * __txn_set_tx_timestamp --
 *	Set the transaction recovery timestamp.
 *
 * PUBLIC: int __txn_set_tx_timestamp __P((DB_ENV *, time_t *));
 */
int
__txn_set_tx_timestamp(dbenv, timestamp)
	DB_ENV *dbenv;
	time_t *timestamp;
{
	ENV *env;

	env = dbenv->env;

	ENV_ILLEGAL_AFTER_OPEN(env, "DB_ENV->set_tx_timestamp");

	dbenv->tx_timestamp = *timestamp;
	return (0);
}
