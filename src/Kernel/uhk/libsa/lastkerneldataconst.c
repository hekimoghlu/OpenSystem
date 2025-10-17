/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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
#include <mach/vm_param.h>

/*
 * This file is compiled and linked to be the last .o of the __const section
 * of the __DATA segment (see MakeInc.kernel, lastkernelconstructor is placed
 * in the __LAST segment.)
 *
 * This blank page allows us to safely map the const section RO while the rest
 * of __DATA is RW. This is needed since ld has no way of specifying section size
 * alignment and no straight forward way to specify section ordering.
 */

#if defined(__arm64__)
/* PAGE_SIZE on ARM64 is an expression derived from a non-const global variable */
#define PAD_SIZE        PAGE_MAX_SIZE
#else
#define PAD_SIZE        PAGE_SIZE
#endif

static const uint8_t __attribute__((section("__DATA,__const"))) data_const_padding[PAD_SIZE] = {[0 ... PAD_SIZE - 1] = 0xFF};
const vm_offset_t    __attribute__((section("__DATA,__data")))  _lastkerneldataconst         = (vm_offset_t)&data_const_padding[0];
const vm_size_t      __attribute__((section("__DATA,__data")))  _lastkerneldataconst_padsize = sizeof(data_const_padding);
