/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#include "dbinc/db_swap.h"
#include "dbinc/btree.h"

/*
 * __bam_pgin --
 *	Convert host-specific page layout from the host-independent format
 *	stored on disk.
 *
 * PUBLIC: int __bam_pgin __P((DB *, db_pgno_t, void *, DBT *));
 */
int
__bam_pgin(dbp, pg, pp, cookie)
	DB *dbp;
	db_pgno_t pg;
	void *pp;
	DBT *cookie;
{
	DB_PGINFO *pginfo;
	PAGE *h;

	pginfo = (DB_PGINFO *)cookie->data;
	if (!F_ISSET(pginfo, DB_AM_SWAP))
		return (0);

	h = pp;
	return (TYPE(h) == P_BTREEMETA ?  __bam_mswap(dbp->env, pp) :
	    __db_byteswap(dbp, pg, pp, pginfo->db_pagesize, 1));
}

/*
 * __bam_pgout --
 *	Convert host-specific page layout to the host-independent format
 *	stored on disk.
 *
 * PUBLIC: int __bam_pgout __P((DB *, db_pgno_t, void *, DBT *));
 */
int
__bam_pgout(dbp, pg, pp, cookie)
	DB *dbp;
	db_pgno_t pg;
	void *pp;
	DBT *cookie;
{
	DB_PGINFO *pginfo;
	PAGE *h;

	pginfo = (DB_PGINFO *)cookie->data;
	if (!F_ISSET(pginfo, DB_AM_SWAP))
		return (0);

	h = pp;
	return (TYPE(h) == P_BTREEMETA ?  __bam_mswap(dbp->env, pp) :
	    __db_byteswap(dbp, pg, pp, pginfo->db_pagesize, 0));
}

/*
 * __bam_mswap --
 *	Swap the bytes on the btree metadata page.
 *
 * PUBLIC: int __bam_mswap __P((ENV *, PAGE *));
 */
int
__bam_mswap(env, pg)
	ENV *env;
	PAGE *pg;
{
	u_int8_t *p;

	COMPQUIET(env, NULL);

	__db_metaswap(pg);
	p = (u_int8_t *)pg + sizeof(DBMETA);

	p += sizeof(u_int32_t);	/* unused */
	SWAP32(p);		/* minkey */
	SWAP32(p);		/* re_len */
	SWAP32(p);		/* re_pad */
	SWAP32(p);		/* root */
	p += 92 * sizeof(u_int32_t); /* unused */
	SWAP32(p);		/* crypto_magic */

	return (0);
}
