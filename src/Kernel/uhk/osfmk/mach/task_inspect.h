/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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
#ifndef MACH_TASK_INSPECT_H
#define MACH_TASK_INSPECT_H

#include <stdint.h>
#include <mach/vm_types.h>

/*
 * XXX These interfaces are still in development -- they are subject to change
 * without notice.
 */

typedef natural_t task_inspect_flavor_t;

enum task_inspect_flavor {
	TASK_INSPECT_BASIC_COUNTS = 1,
};

struct task_inspect_basic_counts {
	uint64_t instructions;
	uint64_t cycles;
};
#define TASK_INSPECT_BASIC_COUNTS_COUNT \
	(sizeof(struct task_inspect_basic_counts) / sizeof(natural_t))
typedef struct task_inspect_basic_counts task_inspect_basic_counts_data_t;
typedef struct task_inspect_basic_counts *task_inspect_basic_counts_t;

typedef integer_t *task_inspect_info_t;

#endif /* !defined(MACH_TASK_INSPECT_H) */
