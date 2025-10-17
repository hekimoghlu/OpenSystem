/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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
/*
 * Copyright 2006 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * Routines for manipulating iidesc_t structures
 */

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "ctftools.h"
#include "memory.h"
#include "list.h"
#include "hash.h"

typedef struct iidesc_find {
	iidesc_t *iif_tgt;
	iidesc_t *iif_ret;
} iidesc_find_t;

iidesc_t *
iidesc_new(atom_t *name)
{
	iidesc_t *ii;

	ii = xcalloc(sizeof (iidesc_t));
	ii->ii_name = name;

	return (ii);
}

int
iidesc_hash(int nbuckets, void *arg)
{
	iidesc_t *ii = arg;
	return atom_hash(ii->ii_name) % nbuckets;
}

static int
iidesc_cmp(iidesc_t *src, iidesc_find_t *find)
{
	iidesc_t *tgt = find->iif_tgt;

	if (src->ii_type != tgt->ii_type ||
	    src->ii_name != tgt->ii_name)
		return (0);

	find->iif_ret = src;

	return (-1);
}

void
iidesc_add(hash_t *hash, iidesc_t *new)
{
	iidesc_find_t find;

	find.iif_tgt = new;
	find.iif_ret = NULL;

	(void) hash_match(hash, new, (int (*)())iidesc_cmp, &find);

	if (find.iif_ret != NULL) {
		iidesc_t *old = find.iif_ret;
		iidesc_t tmp;
		/* replacing existing one */
		bcopy(old, &tmp, sizeof (tmp));
		bcopy(new, old, sizeof (*old));
		bcopy(&tmp, new, sizeof (*new));

		iidesc_free(new, NULL);
		return;
	}

	hash_add(hash, new);
}

void
iter_iidescs_by_name(tdata_t *td, const char *name,
    int (*func)(iidesc_t *, void *), void *data)
{
	iidesc_t tmpdesc;
	bzero(&tmpdesc, sizeof (iidesc_t));
	tmpdesc.ii_name = atom_get(name);
	(void) hash_match(td->td_iihash, &tmpdesc, (int (*)())func, data);
}

iidesc_t *
iidesc_dup(iidesc_t *src)
{
	iidesc_t *tgt;

	tgt = xmalloc(sizeof (iidesc_t));
	bcopy(src, tgt, sizeof (iidesc_t));

	tgt->ii_name = src->ii_name;
	tgt->ii_owner = src->ii_owner;

	if (tgt->ii_nargs) {
		tgt->ii_args = xmalloc(sizeof (tdesc_t *) * tgt->ii_nargs);
		bcopy(src->ii_args, tgt->ii_args,
		    sizeof (tdesc_t *) * tgt->ii_nargs);
	}

	return (tgt);
}

iidesc_t *
iidesc_dup_rename(iidesc_t *src, char const *name, char const *owner)
{
	iidesc_t *tgt = iidesc_dup(src);

	tgt->ii_name = atom_get(name);
	tgt->ii_owner = atom_get(owner);

	return (tgt);
}

/*ARGSUSED*/
void
iidesc_free(iidesc_t *idp, void *private)
{
	if (idp->ii_nargs)
		free(idp->ii_args);
	free(idp);
}

int
iidesc_dump(iidesc_t *ii)
{
	printf("type: %d  name %s\n", ii->ii_type,
	    atom_pretty(ii->ii_name, "(anon)"));

	return (0);
}

int
iidesc_count_type(void *data, void *private)
{
	iidesc_t *ii = data;
	iitype_t match = (iitype_t)private;

	return (ii->ii_type == match);
}

void
iidesc_stats(hash_t *ii)
{
	printf("GFun: %5d SFun: %5d GVar: %5d SVar: %5d T %5d SOU: %5d\n",
	    hash_iter(ii, iidesc_count_type, (void *)II_GFUN),
	    hash_iter(ii, iidesc_count_type, (void *)II_SFUN),
	    hash_iter(ii, iidesc_count_type, (void *)II_GVAR),
	    hash_iter(ii, iidesc_count_type, (void *)II_SVAR),
	    hash_iter(ii, iidesc_count_type, (void *)II_TYPE),
	    hash_iter(ii, iidesc_count_type, (void *)II_SOU));
}
