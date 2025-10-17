/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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
#ifndef _UAPI__ASM_ARM_UNISTD_H
#define _UAPI__ASM_ARM_UNISTD_H
#define __NR_OABI_SYSCALL_BASE 0x900000
#define __NR_SYSCALL_MASK 0x0fffff
#define __NR_SYSCALL_BASE 0
#include <asm/unistd-eabi.h>
#define __NR_sync_file_range2 __NR_arm_sync_file_range
#define __ARM_NR_BASE (__NR_SYSCALL_BASE + 0x0f0000)
#define __ARM_NR_breakpoint (__ARM_NR_BASE + 1)
#define __ARM_NR_cacheflush (__ARM_NR_BASE + 2)
#define __ARM_NR_usr26 (__ARM_NR_BASE + 3)
#define __ARM_NR_usr32 (__ARM_NR_BASE + 4)
#define __ARM_NR_set_tls (__ARM_NR_BASE + 5)
#define __ARM_NR_get_tls (__ARM_NR_BASE + 6)
#endif
