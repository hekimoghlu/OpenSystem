/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
 * If the library wasn't compiled with XA support, various routines
 * aren't available.  Stub them here, returning an appropriate error.
 */
static int __db_noxa __P((DB_ENV *));

/*
 * __db_noxa --
 *	Error when a Berkeley DB build doesn't include XA support.
 */
static int
__db_noxa(dbenv)
	DB_ENV *dbenv;
{
	__db_errx(dbenv->env,
	    "library build did not include support for XA");
	return (DB_OPNOTSUP);
}

int
__db_xa_create(dbp)
	DB *dbp;
{
	return (__db_noxa(dbp->dbenv));
}
