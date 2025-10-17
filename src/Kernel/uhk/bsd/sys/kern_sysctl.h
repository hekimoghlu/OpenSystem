/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#ifndef _KERN_SYSCTL_H_
#define _KERN_SYSCTL_H_

#include <mach/mach_types.h>

typedef struct _vm_object_query_data_ vm_object_query_data_t;
typedef struct _vm_object_query_data_ *vm_object_query_t;

struct _vm_object_query_data_ {
	vm_object_id_t object_id;
	mach_vm_size_t virtual_size;
	mach_vm_size_t resident_size;
	mach_vm_size_t wired_size;
	mach_vm_size_t reusable_size;
	mach_vm_size_t compressed_size;
	struct {
		uint64_t vo_no_footprint : 1; /* object not included in footprint */
		uint64_t vo_ledger_tag   : 3; /* object ledger tag */
		uint64_t purgable        : 2; /* object "purgable" state #defines */
	};
};

typedef struct _vmobject_list_output_ vmobject_list_output_data_t;
typedef struct _vmobject_list_output_ *vmobject_list_output_t;

struct _vmobject_list_output_ {
	uint64_t entries;
	vm_object_query_data_t data[0];
};
#endif /* _KERN_SYSCTL_H_ */
