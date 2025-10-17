/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#ifndef _KCOV_STKSZ_DATA_H_
#define _KCOV_STKSZ_DATA_H_

#include <stdbool.h>
#include <mach/vm_types.h>

#if KERNEL_PRIVATE

#if CONFIG_STKSZ

/*
 * Stack size monitor per-cpu data.
 */
typedef struct kcov_stksz_thread {
	vm_offset_t    kst_stack;       /* thread stack override */
	uintptr_t      kst_pc;          /* last seen program counter */
	uint32_t       kst_stksz;       /* last seen stack size */
	uint32_t       kst_stksz_prev;  /* previous known stack size */
	bool           kst_th_above;    /* threshold */
} kcov_stksz_thread_t;

#endif /* CONFIG_STKSZ */

#endif /* KERNEL_PRIVATE */

#endif /* _KCOV_STKSZ_DATA_H_ */
