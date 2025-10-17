/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#ifndef _I386_MEMORY_TYPES_H_
#define _I386_MEMORY_TYPES_H_

#define VM_WIMG_COPYBACK                  VM_MEM_COHERENT
#define VM_WIMG_COPYBACKLW                VM_WIMG_COPYBACK
#define VM_WIMG_DEFAULT                   VM_MEM_COHERENT
/* ?? intel ?? */
#define VM_WIMG_IO                        (VM_MEM_COHERENT |      \
	                                  VM_MEM_NOT_CACHEABLE | VM_MEM_GUARDED)
#define VM_WIMG_POSTED                    VM_WIMG_IO
#define VM_WIMG_POSTED_REORDERED          VM_WIMG_IO
#define VM_WIMG_POSTED_COMBINED_REORDERED VM_WIMG_IO
#define VM_WIMG_WTHRU                     (VM_MEM_WRITE_THROUGH | VM_MEM_COHERENT | VM_MEM_GUARDED)
/* write combining mode, aka store gather */
#define VM_WIMG_WCOMB                     (VM_MEM_NOT_CACHEABLE | VM_MEM_COHERENT)
#define VM_WIMG_INNERWBACK       VM_MEM_COHERENT
#define VM_WIMG_RT               VM_WIMG_WCOMB

#endif /* _I386_MEMORY_TYPES_H_ */
