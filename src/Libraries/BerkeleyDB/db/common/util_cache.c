/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
 * __db_util_cache --
 *	Compute if we have enough cache.
 *
 * PUBLIC: int __db_util_cache __P((DB *, u_int32_t *, int *));
 */
int
__db_util_cache(dbp, cachep, resizep)
	DB *dbp;
	u_int32_t *cachep;
	int *resizep;
{
	u_int32_t pgsize;
	int ret;

	/* Get the current page size. */
	if ((ret = dbp->get_pagesize(dbp, &pgsize)) != 0)
		return (ret);

	/*
	 * The current cache size is in cachep.  If it's insufficient, set the
	 * the memory referenced by resizep to 1 and set cachep to the minimum
	 * size needed.
	 *
	 * Make sure our current cache is big enough.  We want at least
	 * DB_MINPAGECACHE pages in the cache.
	 */
	if ((*cachep / pgsize) < DB_MINPAGECACHE) {
		*resizep = 1;
		*cachep = pgsize * DB_MINPAGECACHE;
	} else
		*resizep = 0;

	return (0);
}
