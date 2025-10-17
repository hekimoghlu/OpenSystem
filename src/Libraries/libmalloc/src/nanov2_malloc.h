/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#ifndef __NANOV2_MALLOC_H
#define __NANOV2_MALLOC_H

#include <malloc/_ptrcheck.h>
__ptrcheck_abi_assume_single()

// Forward declaration for the nanozonev2 structure.
typedef struct nanozonev2_s nanozonev2_t;

MALLOC_NOEXPORT
void
nanov2_init(const char * __null_terminated * __null_terminated envp, const char * __null_terminated * __null_terminated apple, const char *bootargs);

MALLOC_NOEXPORT
void
nanov2_configure(void);

MALLOC_NOEXPORT
malloc_zone_t *
nanov2_create_zone(malloc_zone_t *helper_zone, unsigned debug_flags);

MALLOC_NOEXPORT
void
nanov2_forked_zone(nanozonev2_t *nanozone);

#endif // __NANOV2_MALLOC_H
