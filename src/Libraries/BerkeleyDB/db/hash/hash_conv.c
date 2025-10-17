/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 1, 2023.
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
#include "dbinc/hash.h"

/*
 * __ham_pgin --
 *	Convert host-specific page layout from the host-independent format
 *	stored on disk.
 *
 * PUBLIC: int __ham_pgin __P((DB *, db_pgno_t, void *, DBT *));
 */
int
__ham_pgin(dbp, pg, pp, cookie)
	DB *dbp;
	db_pgno_t pg;
	void *pp;
	DBT *cookie;
{
	DB_PGINFO *pginfo;
	PAGE *h;

	h = pp;
	pginfo = (DB_PGINFO *)cookie->data;

	/*
	 * The hash access method does blind reads of pages, causing them
	 * to be created.  If the type field isn't set it's one of them,
	 * initialize the rest of the page and return.
	 */
	if (h->type != P_HASHMETA && h->pgno == PGNO_INVALID) {
		P_INIT(pp, (db_indx_t)pginfo->db_pagesize,
		    pg, PGNO_INVALID, PGNO_INVALID, 0, P_HASH);
		return (0);
	}

	if (!F_ISSET(pginfo, DB_AM_SWAP))
		return (0);

	return (h->type == P_HASHMETA ? __ham_mswap(dbp->env, pp) :
	    __db_byteswap(dbp, pg, pp, pginfo->db_pagesize, 1));
}

/*
 * __ham_pgout --
 *	Convert host-specific page layout to the host-independent format
 *	stored on disk.
 *
 * PUBLIC: int __ham_pgout __P((DB *, db_pgno_t, void *, DBT *));
 */
int
__ham_pgout(dbp, pg, pp, cookie)
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
	return (h->type == P_HASHMETA ?  __ham_mswap(dbp->env, pp) :
	    __db_byteswap(dbp, pg, pp, pginfo->db_pagesize, 0));
}

/*
 * __ham_mswap --
 *	Swap the bytes on the hash metadata page.
 *
 * PUBLIC: int __ham_mswap __P((ENV *, void *));
 */
int
__ham_mswap(env, pg)
	ENV *env;
	void *pg;
{
	u_int8_t *p;
	int i;

	COMPQUIET(env, NULL);

	__db_metaswap(pg);
	p = (u_int8_t *)pg + sizeof(DBMETA);

	SWAP32(p);		/* max_bucket */
	SWAP32(p);		/* high_mask */
	SWAP32(p);		/* low_mask */
	SWAP32(p);		/* ffactor */
	SWAP32(p);		/* nelem */
	SWAP32(p);		/* h_charkey */
	for (i = 0; i < NCACHED; ++i)
		SWAP32(p);	/* spares */
	p += 59 * sizeof(u_int32_t); /* unused */
	SWAP32(p);		/* crypto_magic */
	return (0);
}
