/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#ifndef PANICHOOKS_H_
#define PANICHOOKS_H_

#if XNU_KERNEL_PRIVATE

#include <stdint.h>
#include <mach/i386/boolean.h>

typedef struct {
	uint64_t        opaque[6];
} panic_hook_t;

typedef void (*panic_hook_fn_t)(panic_hook_t *);

void panic_hooks_init(void);
void panic_check_hook(void);

void panic_hook(panic_hook_t *hook, panic_hook_fn_t hook_fn);
void panic_unhook(panic_hook_t *hook);
void panic_dump_mem(const void *addr, int len);

typedef struct panic_phys_range {
	uint32_t type;
	uint64_t phys_start;
	uint64_t len;
} panic_phys_range_t;

boolean_t panic_phys_range_before(const void *addr, uint64_t *pphys,
    panic_phys_range_t *range);

#endif // XNU_KERNEL_PRIVATE

#endif // PANICHOOKS_H_
