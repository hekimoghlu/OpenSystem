/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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
#pragma once

#include <mach/port.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
	kHV_ION_NONE             = (0u << 0),
	kHV_ION_ANY_VALUE        = (1u << 1),
	kHV_ION_ANY_SIZE         = (1u << 2),
	kHV_ION_EXIT_FULL        = (1u << 3),
};

#ifdef KERNEL_PRIVATE

typedef struct {
	mach_msg_header_t header;
	uint64_t addr;
	uint64_t size;
	uint64_t value;
} hv_ion_message_t;

typedef struct {
	uint64_t addr;
	uint64_t size;
	uint64_t value;
	uint32_t port_name;
	uint32_t flags;
} hv_ion_t;

typedef struct hv_ion_grp hv_ion_grp_t;

extern kern_return_t hv_io_notifier_grp_add(hv_ion_grp_t *grp, const hv_ion_t *);
extern kern_return_t hv_io_notifier_grp_remove(hv_ion_grp_t *, const hv_ion_t *);
extern kern_return_t hv_io_notifier_grp_fire(hv_ion_grp_t *, uint64_t, size_t, uint64_t);
extern kern_return_t hv_io_notifier_grp_alloc(hv_ion_grp_t **);
extern void hv_io_notifier_grp_free(hv_ion_grp_t **);

#endif /* KERNEL_PRIVATE */

#ifdef __cplusplus
}
#endif
