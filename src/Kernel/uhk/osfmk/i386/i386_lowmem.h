/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
#ifndef _I386_LOWMEM_H_
#define _I386_LOWMEM_H_


#ifdef __APPLE_API_PRIVATE

/*
 * The kernel better be statically linked at VM_MIN_KERNEL_ADDRESS + 0x100000
 */
#define I386_KERNEL_IMAGE_BASE_PAGE     0x100

/* For K64, only 3 pages are reserved
 * - physical page zero, a gap page, and then real-mode-bootstrap/lowGlo.
 * Note that the kernel virtual address KERNEL_BASE+0x2000 is re-mapped
 * to the low globals and that physical page, 0x2000, is used by the bootstrap.
 */
#define I386_LOWMEM_RESERVED            3

#endif /* __APPLE_API_PRIVATE */

#endif /* !_I386_LOWMEM_H_ */
