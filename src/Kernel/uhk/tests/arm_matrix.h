/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#ifndef __ARM_MATRIX_H
#define __ARM_MATRIX_H

#include <mach/mach_types.h>
#include <mach/thread_status.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

struct arm_matrix_operations {
	const char *name;

	size_t (*data_size)(void);
	void *(*alloc_data)(void);

	bool (*is_available)(void);
	void (*start)(void);
	void (*stop)(void);

	void (*load_one_vector)(const void *);
	void (*load_data)(const void *);
	void (*store_data)(void *);

	kern_return_t (*thread_get_state)(thread_act_t, void *);
	kern_return_t (*thread_set_state)(thread_act_t, const void *);
};

extern const struct arm_matrix_operations sme_operations;

#endif /* __ARM_MATRIX_H */
