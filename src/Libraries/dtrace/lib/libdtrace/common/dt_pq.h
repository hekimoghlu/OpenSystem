/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
 * Copyright (c) 2012 by Delphix. All rights reserved.
 */

#ifndef	_DT_PQ_H
#define	_DT_PQ_H

#include <dtrace.h>

#ifdef	__cplusplus
extern "C" {
#endif

typedef uint64_t (*dt_pq_value_f)(void *, void *);

typedef struct dt_pq {
	dtrace_hdl_t *dtpq_hdl;		/* dtrace handle */
	void **dtpq_items;		/* array of elements */
	uint_t dtpq_size;		/* count of allocated elements */
	uint_t dtpq_last;		/* next free slot */
	dt_pq_value_f dtpq_value;	/* callback to get the value */
	void *dtpq_arg;			/* callback argument */
} dt_pq_t;

extern dt_pq_t *dt_pq_init(dtrace_hdl_t *, uint_t size, dt_pq_value_f, void *);
extern void dt_pq_fini(dt_pq_t *);

extern void dt_pq_insert(dt_pq_t *, void *);
extern void *dt_pq_pop(dt_pq_t *);
extern void *dt_pq_walk(dt_pq_t *, uint_t *);

#ifdef	__cplusplus
}
#endif

#endif	/* _DT_PQ_H */

