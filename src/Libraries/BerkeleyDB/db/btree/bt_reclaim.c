/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#include "dbinc/btree.h"
#include "dbinc/lock.h"

/*
 * __bam_reclaim --
 *	Free a database.
 *
 * PUBLIC: int __bam_reclaim __P((DB *, DB_THREAD_INFO *, DB_TXN *));
 */
int
__bam_reclaim(dbp, ip, txn)
	DB *dbp;
	DB_THREAD_INFO *ip;
	DB_TXN *txn;
{
	DBC *dbc;
	DB_LOCK meta_lock;
	int ret, t_ret;

	/* Acquire a cursor. */
	if ((ret = __db_cursor(dbp, ip, txn, &dbc, 0)) != 0)
		return (ret);

	/* Write lock the metapage for deallocations. */
	if ((ret = __db_lget(dbc,
	    0, PGNO_BASE_MD, DB_LOCK_WRITE, 0, &meta_lock)) != 0)
		goto err;

	/* Avoid locking every page, we have the handle locked exclusive. */
	F_SET(dbc, DBC_DONTLOCK);

	/* Walk the tree, freeing pages. */
	ret = __bam_traverse(dbc,
	    DB_LOCK_WRITE, dbc->internal->root, __db_reclaim_callback, NULL);

	if ((t_ret = __TLPUT(dbc, meta_lock)) != 0 && ret == 0)
		ret = t_ret;

	/* Discard the cursor. */
err:	if ((t_ret = __dbc_close(dbc)) != 0 && ret == 0)
		ret = t_ret;

	return (ret);
}

/*
 * __bam_truncate --
 *	Truncate a database.
 *
 * PUBLIC: int __bam_truncate __P((DBC *, u_int32_t *));
 */
int
__bam_truncate(dbc, countp)
	DBC *dbc;
	u_int32_t *countp;
{
	u_int32_t count;
	int ret;

	count = 0;

	/* Walk the tree, freeing pages. */
	ret = __bam_traverse(dbc,
	    DB_LOCK_WRITE, dbc->internal->root, __db_truncate_callback, &count);

	if (countp != NULL)
		*countp = count;

	return (ret);
}
