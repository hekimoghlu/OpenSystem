/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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

// It's private, and should never be called,
// but some compilers need it resolved
//
DbLogc::~DbLogc()
{
}

// The name _flags prevents a name clash with __db_log_cursor::flags
int DbLogc::close(u_int32_t _flags)
{
	DB_LOGC *logc = this;
	int ret;
	DbEnv *dbenv2 = DbEnv::get_DbEnv(logc->env->dbenv);

	ret = logc->close(logc, _flags);

	if (!DB_RETOK_STD(ret))
		DB_ERROR(dbenv2, "DbLogc::close", ret, ON_ERROR_UNKNOWN);

	return (ret);
}

// The name _flags prevents a name clash with __db_log_cursor::flags
int DbLogc::get(DbLsn *get_lsn, Dbt *data, u_int32_t _flags)
{
	DB_LOGC *logc = this;
	int ret;

	ret = logc->get(logc, get_lsn, data, _flags);

	if (!DB_RETOK_LGGET(ret)) {
		if (ret == DB_BUFFER_SMALL)
			DB_ERROR_DBT(DbEnv::get_DbEnv(logc->env->dbenv),
				"DbLogc::get", data, ON_ERROR_UNKNOWN);
		else
			DB_ERROR(DbEnv::get_DbEnv(logc->env->dbenv),
				"DbLogc::get", ret, ON_ERROR_UNKNOWN);
	}

	return (ret);
}

// The name _flags prevents a name clash with __db_log_cursor::flags
int DbLogc::version(u_int32_t *versionp, u_int32_t _flags)
{
	DB_LOGC *logc = this;
	int ret;

	ret = logc->version(logc, versionp, _flags);

	if (!DB_RETOK_LGGET(ret))
		DB_ERROR(DbEnv::get_DbEnv(logc->env->dbenv),
			"DbLogc::version", ret, ON_ERROR_UNKNOWN);

	return (ret);
}
