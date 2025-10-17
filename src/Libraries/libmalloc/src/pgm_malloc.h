/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#ifndef _PGM_MALLOC_H_
#define _PGM_MALLOC_H_

#include "base.h"
#include "malloc/malloc.h"
#include <stdbool.h>

MALLOC_NOEXPORT
void
pgm_init_config(bool internal_build);

MALLOC_NOEXPORT
bool
pgm_should_enable(void);

MALLOC_NOEXPORT
malloc_zone_t *
pgm_create_zone(malloc_zone_t *wrapped_zone);

MALLOC_NOEXPORT
void
pgm_thread_set_disabled(bool disabled);

#endif // _PGM_MALLOC_H_
