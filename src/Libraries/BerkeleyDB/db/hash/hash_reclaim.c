/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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
#include "dbinc/hash.h"

/*
 * __ham_reclaim --
 *	Reclaim the pages from a subdatabase and return them to the
 * parent free list.  For now, we link each freed page on the list
 * separately.  If people really store hash databases in subdatabases
 * and do a lot of creates and deletes, this is going to be a problem,
 * because hash needs chunks of contiguous storage.  We may eventually
 * need to go to a model where we maintain the free list with chunks of
 * contiguous pages as well.
 *
 * PUBLIC: int __ham_reclaim __P((DB *, DB_THREAD_INFO *, DB_TXN *txn));
 */
int
__ham_reclaim(dbp, ip, txn)
	DB *dbp;
	DB_THREAD_INFO *ip;
	DB_TXN *txn;
{
	DBC *dbc;
	HASH_CURSOR *hcp;
	int ret;

	/* Open up a cursor that we'll use for traversing. */
	if ((ret = __db_cursor(dbp, ip, txn, &dbc, 0)) != 0)
		return (ret);
	hcp = (HASH_CURSOR *)dbc->internal;

	if ((ret = __ham_get_meta(dbc)) != 0)
		goto err;

	/* Write lock the metapage for deallocations. */
	if ((ret = __ham_dirty_meta(dbc, 0)) != 0)
		goto err;

	/* Avoid locking every page, we have the handle locked exclusive. */
	F_SET(dbc, DBC_DONTLOCK);

	if ((ret = __ham_traverse(dbc,
	    DB_LOCK_WRITE, __db_reclaim_callback, NULL, 1)) != 0)
		goto err;
	if ((ret = __dbc_close(dbc)) != 0)
		goto err;
	if ((ret = __ham_release_meta(dbc)) != 0)
		goto err;
	return (0);

err:	if (hcp->hdr != NULL)
		(void)__ham_release_meta(dbc);
	(void)__dbc_close(dbc);
	return (ret);
}

/*
 * __ham_truncate --
 *	Reclaim the pages from a subdatabase and return them to the
 * parent free list.
 *
 * PUBLIC: int __ham_truncate __P((DBC *, u_int32_t *));
 */
int
__ham_truncate(dbc, countp)
	DBC *dbc;
	u_int32_t *countp;
{
	u_int32_t count;
	int ret, t_ret;

	if ((ret = __ham_get_meta(dbc)) != 0)
		return (ret);

	count = 0;

	ret = __ham_traverse(dbc,
	    DB_LOCK_WRITE, __db_truncate_callback, &count, 1);

	if ((t_ret = __ham_release_meta(dbc)) != 0 && ret == 0)
		ret = t_ret;

	if (countp != NULL)
		*countp = count;
	return (ret);
}
