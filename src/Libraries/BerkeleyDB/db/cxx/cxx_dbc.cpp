/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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

#include "db_cxx.h"
#include "dbinc/cxx_int.h"

#include "dbinc/db_page.h"
#include "dbinc_auto/db_auto.h"
#include "dbinc_auto/crdel_auto.h"
#include "dbinc/db_dispatch.h"
#include "dbinc_auto/db_ext.h"
#include "dbinc_auto/common_ext.h"

// Helper macro for simple methods that pass through to the
// underlying C method. It may return an error or raise an exception.
// Note this macro expects that input _argspec is an argument
// list element (e.g., "char *arg") and that _arglist is the arguments
// that should be passed through to the C method (e.g., "(db, arg)")
//
#define	DBC_METHOD(_name, _argspec, _arglist, _retok)			\
int Dbc::_name _argspec							\
{									\
	int ret;							\
	DBC *dbc = this;						\
									\
	ret = dbc->_name _arglist;					\
	if (!_retok(ret))						\
		DB_ERROR(DbEnv::get_DbEnv(dbc->dbenv),			\
			"Dbc::" # _name, ret, ON_ERROR_UNKNOWN);	\
	return (ret);							\
}

// It's private, and should never be called, but VC4.0 needs it resolved
//
Dbc::~Dbc()
{
}

DBC_METHOD(close, (void), (dbc), DB_RETOK_STD)
DBC_METHOD(count, (db_recno_t *countp, u_int32_t _flags),
    (dbc, countp, _flags), DB_RETOK_STD)
DBC_METHOD(del, (u_int32_t _flags),
    (dbc, _flags), DB_RETOK_DBCDEL)

int Dbc::dup(Dbc** cursorp, u_int32_t _flags)
{
	int ret;
	DBC *dbc = this;
	DBC *new_cursor = 0;

	ret = dbc->dup(dbc, &new_cursor, _flags);

	if (DB_RETOK_STD(ret))
		// The following cast implies that Dbc can be no larger than DBC
		*cursorp = (Dbc*)new_cursor;
	else
		DB_ERROR(DbEnv::get_DbEnv(dbc->dbenv),
			"Dbc::dup", ret, ON_ERROR_UNKNOWN);

	return (ret);
}

int Dbc::get(Dbt* key, Dbt *data, u_int32_t _flags)
{
	int ret;
	DBC *dbc = this;

	ret = dbc->get(dbc, key, data, _flags);

	if (!DB_RETOK_DBCGET(ret)) {
		if (ret == DB_BUFFER_SMALL && DB_OVERFLOWED_DBT(key))
			DB_ERROR_DBT(DbEnv::get_DbEnv(dbc->dbenv),
				"Dbc::get", key, ON_ERROR_UNKNOWN);
		else if (ret == DB_BUFFER_SMALL && DB_OVERFLOWED_DBT(data))
			DB_ERROR_DBT(DbEnv::get_DbEnv(dbc->dbenv),
				"Dbc::get", data, ON_ERROR_UNKNOWN);
		else
			DB_ERROR(DbEnv::get_DbEnv(dbc->dbenv),
				"Dbc::get", ret, ON_ERROR_UNKNOWN);
	}

	return (ret);
}

int Dbc::pget(Dbt* key, Dbt *pkey, Dbt *data, u_int32_t _flags)
{
	int ret;
	DBC *dbc = this;

	ret = dbc->pget(dbc, key, pkey, data, _flags);

	/* Logic is the same as for Dbc::get - reusing macro. */
	if (!DB_RETOK_DBCGET(ret)) {
		if (ret == DB_BUFFER_SMALL && DB_OVERFLOWED_DBT(key))
			DB_ERROR_DBT(DbEnv::get_DbEnv(dbc->dbenv),
				"Dbc::pget", key, ON_ERROR_UNKNOWN);
		else if (ret == DB_BUFFER_SMALL && DB_OVERFLOWED_DBT(data))
			DB_ERROR_DBT(DbEnv::get_DbEnv(dbc->dbenv),
				"Dbc::pget", data, ON_ERROR_UNKNOWN);
		else
			DB_ERROR(DbEnv::get_DbEnv(dbc->dbenv),
				"Dbc::pget", ret, ON_ERROR_UNKNOWN);
	}

	return (ret);
}

DBC_METHOD(put, (Dbt* key, Dbt *data, u_int32_t _flags),
    (dbc, key, data, _flags), DB_RETOK_DBCPUT)
DBC_METHOD(get_priority, (DB_CACHE_PRIORITY *priorityp),
    (dbc, priorityp), DB_RETOK_STD)
DBC_METHOD(set_priority, (DB_CACHE_PRIORITY pri), (dbc, pri), DB_RETOK_STD)
