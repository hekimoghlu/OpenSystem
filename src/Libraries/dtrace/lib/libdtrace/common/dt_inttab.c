/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#include <assert.h>

#include <dt_inttab.h>
#include <dt_impl.h>

dt_inttab_t *
dt_inttab_create(dtrace_hdl_t *dtp)
{
	uint_t len = _dtrace_intbuckets;
	dt_inttab_t *ip;

	assert((len & (len - 1)) == 0);

	if ((ip = dt_zalloc(dtp, sizeof (dt_inttab_t))) == NULL ||
	    (ip->int_hash = dt_zalloc(dtp, sizeof (void *) * len)) == NULL) {
		dt_free(dtp, ip);
		return (NULL);
	}

	ip->int_hdl = dtp;
	ip->int_hashlen = len;

	return (ip);
}

void
dt_inttab_destroy(dt_inttab_t *ip)
{
	dt_inthash_t *hp, *np;

	for (hp = ip->int_head; hp != NULL; hp = np) {
		np = hp->inh_next;
		dt_free(ip->int_hdl, hp);
	}

	dt_free(ip->int_hdl, ip->int_hash);
	dt_free(ip->int_hdl, ip);
}

int
dt_inttab_insert(dt_inttab_t *ip, uint64_t value, uint_t flags)
{
	uint_t h = value & (ip->int_hashlen - 1);
	dt_inthash_t *hp;

	if (flags & DT_INT_SHARED) {
		for (hp = ip->int_hash[h]; hp != NULL; hp = hp->inh_hash) {
			if (hp->inh_value == value && hp->inh_flags == flags)
				return (hp->inh_index);
		}
	}

	if ((hp = dt_alloc(ip->int_hdl, sizeof (dt_inthash_t))) == NULL)
		return (-1);

	hp->inh_hash = ip->int_hash[h];
	hp->inh_next = NULL;
	hp->inh_value = value;
	hp->inh_index = ip->int_index++;
	hp->inh_flags = flags;

	ip->int_hash[h] = hp;
	ip->int_nelems++;

	if (ip->int_head == NULL)
		ip->int_head = hp;
	else
		ip->int_tail->inh_next = hp;

	ip->int_tail = hp;
	return (hp->inh_index);
}

uint_t
dt_inttab_size(const dt_inttab_t *ip)
{
	return (ip->int_nelems);
}

void
dt_inttab_write(const dt_inttab_t *ip, uint64_t *dst)
{
	const dt_inthash_t *hp;

	for (hp = ip->int_head; hp != NULL; hp = hp->inh_next)
		*dst++ = hp->inh_value;
}
