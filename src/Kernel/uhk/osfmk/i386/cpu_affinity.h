/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
#ifdef KERNEL_PRIVATE
#ifndef _I386_CPU_AFFINITY_H_
#define _I386_CPU_AFFINITY_H_

#include <i386/cpu_topology.h>

typedef struct x86_affinity_set {
	struct x86_affinity_set     *next;/* Forward link */
	struct x86_cpu_cache        *cache;/* The L2 cache concerned */
	processor_set_t             pset;/* The processor set container */
	uint32_t                    num;/* Logical id */
} x86_affinity_set_t;

extern x86_affinity_set_t *x86_affinities;      /* root of all affinities */

extern int              ml_get_max_affinity_sets(void);
extern processor_set_t  ml_affinity_to_pset(uint32_t affinity_num);

#endif /* _I386_CPU_AFFINITY_H_ */
#endif /* KERNEL_PRIVATE */
