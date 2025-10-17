/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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
/* public domain */
/*
 * Physical memory is mapped 1:1 at the bottom of the supervisor address
 * space. Kernel virtual memory space starts from the end of physical memory,
 * up to the on-board devices appearing all over the last 3GB of address space.
 */
#define VM_MIN_KERNEL_ADDRESS	((vaddr_t)0x00000000)
#define VM_MAX_KERNEL_ADDRESS	((vaddr_t)0x40000000)

#include <m88k/vmparam.h>
