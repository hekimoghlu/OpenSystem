/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#ifndef netdissect_alloc_h
#define netdissect_alloc_h

#include <stdarg.h>
#include "netdissect-stdinc.h"
#include "netdissect.h"

typedef struct nd_mem_chunk {
	void *prev_mem_p;
	/* variable size data */
} nd_mem_chunk_t;

void nd_add_alloc_list(netdissect_options *, nd_mem_chunk_t *);
void * nd_malloc(netdissect_options *, size_t);
void nd_free_all(netdissect_options *);

#endif /* netdissect_alloc_h */
