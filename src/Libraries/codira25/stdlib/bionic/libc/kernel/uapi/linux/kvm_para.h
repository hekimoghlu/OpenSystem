/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#ifndef _UAPI__LINUX_KVM_PARA_H
#define _UAPI__LINUX_KVM_PARA_H
#define KVM_ENOSYS 1000
#define KVM_EFAULT EFAULT
#define KVM_EINVAL EINVAL
#define KVM_E2BIG E2BIG
#define KVM_EPERM EPERM
#define KVM_EOPNOTSUPP 95
#define KVM_HC_VAPIC_POLL_IRQ 1
#define KVM_HC_MMU_OP 2
#define KVM_HC_FEATURES 3
#define KVM_HC_PPC_MAP_MAGIC_PAGE 4
#define KVM_HC_KICK_CPU 5
#define KVM_HC_MIPS_GET_CLOCK_FREQ 6
#define KVM_HC_MIPS_EXIT_VM 7
#define KVM_HC_MIPS_CONSOLE_OUTPUT 8
#define KVM_HC_CLOCK_PAIRING 9
#define KVM_HC_SEND_IPI 10
#define KVM_HC_SCHED_YIELD 11
#define KVM_HC_MAP_GPA_RANGE 12
#include <asm/kvm_para.h>
#endif
