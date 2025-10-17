/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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
#ifndef _ARM_MEMORY_TYPES_H_
#define _ARM_MEMORY_TYPES_H_

#include <machine/config.h>

#define VM_MEM_INNER                      0x10
#define VM_MEM_RT                         0x10 // intentionally alias VM_MEM_INNER; will be used with mutually exclusive caching policies
#define VM_MEM_EARLY_ACK                  0x20

#define VM_WIMG_DEFAULT                   (VM_MEM_COHERENT) // 0x2
#define VM_WIMG_COPYBACK                  (VM_MEM_COHERENT) // 0x2
#define VM_WIMG_INNERWBACK                (VM_MEM_COHERENT | VM_MEM_INNER) // 0x12
#define VM_WIMG_IO                        (VM_MEM_COHERENT | VM_MEM_NOT_CACHEABLE | VM_MEM_GUARDED) // 0x7
#define VM_WIMG_POSTED                    (VM_MEM_COHERENT | VM_MEM_NOT_CACHEABLE | VM_MEM_GUARDED | VM_MEM_EARLY_ACK) // 0x27
#define VM_WIMG_WTHRU                     (VM_MEM_WRITE_THROUGH | VM_MEM_COHERENT | VM_MEM_GUARDED) // 0xb
#define VM_WIMG_WCOMB                     (VM_MEM_NOT_CACHEABLE | VM_MEM_COHERENT) // 0x6
#if HAS_UCNORMAL_MEM || APPLEVIRTUALPLATFORM
#define VM_WIMG_RT                        (VM_WIMG_WCOMB | VM_MEM_RT) // 0x16
#else
#define VM_WIMG_RT                        (VM_WIMG_IO | VM_MEM_RT) // 0x17
#endif
#define VM_WIMG_POSTED_REORDERED          (VM_MEM_NOT_CACHEABLE | VM_MEM_COHERENT | VM_MEM_WRITE_THROUGH | VM_MEM_EARLY_ACK) // 0x2e
#define VM_WIMG_POSTED_COMBINED_REORDERED (VM_MEM_NOT_CACHEABLE | VM_MEM_COHERENT | VM_MEM_EARLY_ACK) // 0x26


#endif /* _ARM_MEMORY_TYPES_H_ */
