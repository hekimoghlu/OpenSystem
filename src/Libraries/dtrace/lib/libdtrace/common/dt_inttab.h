/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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

#ifndef	_DT_INTTAB_H
#define	_DT_INTTAB_H

#include <dtrace.h>

#ifdef	__cplusplus
extern "C" {
#endif

typedef struct dt_inthash {
	struct dt_inthash *inh_hash;	/* next dt_inthash in hash chain */
	struct dt_inthash *inh_next;	/* next dt_inthash in output table */
	uint64_t inh_value;		/* value associated with this element */
	uint_t inh_index;		/* index associated with this element */
	uint_t inh_flags;		/* flags (see below) */
} dt_inthash_t;

typedef struct dt_inttab {
	dtrace_hdl_t *int_hdl;		/* pointer back to library handle */
	dt_inthash_t **int_hash;	/* array of hash buckets */
	uint_t int_hashlen;		/* size of hash bucket array */
	uint_t int_nelems;		/* number of elements hashed */
	dt_inthash_t *int_head;		/* head of table in index order */
	dt_inthash_t *int_tail;		/* tail of table in index order */
	uint_t int_index;		/* next index to hand out */
} dt_inttab_t;

#define	DT_INT_PRIVATE	0		/* only a single ref for this entry */
#define	DT_INT_SHARED	1		/* multiple refs can share entry */

extern dt_inttab_t *dt_inttab_create(dtrace_hdl_t *);
extern void dt_inttab_destroy(dt_inttab_t *);
extern int dt_inttab_insert(dt_inttab_t *, uint64_t, uint_t);
extern uint_t dt_inttab_size(const dt_inttab_t *);
extern void dt_inttab_write(const dt_inttab_t *, uint64_t *);

#ifdef	__cplusplus
}
#endif

#endif	/* _DT_INTTAB_H */
