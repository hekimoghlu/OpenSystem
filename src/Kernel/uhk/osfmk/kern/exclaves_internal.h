/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#if CONFIG_EXCLAVES

#pragma once

#include <stdbool.h>

#include <mach/kern_return.h>

__BEGIN_DECLS

/*
 * Used to determine if a given return address is in a specific range when
 * determining where to stick stacks from exclaves and xnu.
 */
static inline bool
exclaves_in_range(uintptr_t addr, uintptr_t start, uintptr_t end)
{
	return addr > start && addr <= end;
}

extern lck_grp_t exclaves_lck_grp;

/*
 * Run the specified thread's scheduling context in exclaves.
 */
extern kern_return_t
exclaves_run(thread_t thread, bool interrupted);

__END_DECLS

#endif /* CONFIG_EXCLAVES */
