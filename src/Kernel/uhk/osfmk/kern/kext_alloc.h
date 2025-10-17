/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#ifndef _KEXT_ALLOC_H_
#define _KEXT_ALLOC_H_

#include <mach/kern_return.h>
#include <mach/vm_types.h>

__BEGIN_DECLS

vm_offset_t get_address_from_kext_map(vm_size_t fsize);

void kext_alloc_init(void);

kern_return_t kext_alloc(vm_offset_t *addr, vm_size_t size, boolean_t fixed);

void kext_free(vm_offset_t addr, vm_size_t size);

kern_return_t kext_receipt(void **addrp, size_t *sizep);

kern_return_t kext_receipt_set_queried(void);

__END_DECLS

#endif /* _KEXT_ALLOC_H_ */
