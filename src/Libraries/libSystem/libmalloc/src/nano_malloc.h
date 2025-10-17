/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#ifndef __NANO_MALLOC_H
#define __NANO_MALLOC_H

// Forward decl for the nanozone.
typedef struct nanozone_s nanozone_t;

MALLOC_NOEXPORT
malloc_zone_t *
nano_create_zone(malloc_zone_t *helper_zone, unsigned debug_flags);

MALLOC_NOEXPORT
void
nano_forked_zone(nanozone_t *nanozone);

MALLOC_NOEXPORT
void
nano_init(const char *envp[], const char *apple[], const char *bootargs);

MALLOC_NOEXPORT
void
nano_configure(void);

#endif // __NANO_MALLOC_H
