/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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

#ifndef _TRAVERSE_H
#define	_TRAVERSE_H

/*
 * Routines used to traverse tdesc trees, invoking user-supplied callbacks
 * as the tree is traversed.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "ctftools.h"

typedef int (*tdtrav_cb_f)(tdesc_t *, tdesc_t **, void *);

typedef struct tdtrav_data {
	int vgen;

	tdtrav_cb_f *firstops;
	tdtrav_cb_f *preops;
	tdtrav_cb_f *postops;

	void *private;
} tdtrav_data_t;

void tdtrav_init(tdtrav_data_t *, int *, tdtrav_cb_f *, tdtrav_cb_f *,
    tdtrav_cb_f *, void *);
int tdtraverse(tdesc_t *, tdesc_t **, tdtrav_data_t *);

int iitraverse(iidesc_t *, int *, tdtrav_cb_f *, tdtrav_cb_f *, tdtrav_cb_f *,
    void *);
int iitraverse_hash(hash_t *, int *, tdtrav_cb_f *, tdtrav_cb_f *,
    tdtrav_cb_f *, void *);
int iitraverse_td(iidesc_t *ii, tdtrav_data_t *);

int tdtrav_assert(tdesc_t *, tdesc_t **, void *);

#ifdef __cplusplus
}
#endif

#endif /* _TRAVERSE_H */
