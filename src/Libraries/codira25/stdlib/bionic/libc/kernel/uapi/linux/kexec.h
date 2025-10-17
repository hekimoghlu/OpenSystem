/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
#ifndef _UAPILINUX_KEXEC_H
#define _UAPILINUX_KEXEC_H
#include <linux/types.h>
#define KEXEC_ON_CRASH 0x00000001
#define KEXEC_PRESERVE_CONTEXT 0x00000002
#define KEXEC_UPDATE_ELFCOREHDR 0x00000004
#define KEXEC_CRASH_HOTPLUG_SUPPORT 0x00000008
#define KEXEC_ARCH_MASK 0xffff0000
#define KEXEC_FILE_UNLOAD 0x00000001
#define KEXEC_FILE_ON_CRASH 0x00000002
#define KEXEC_FILE_NO_INITRAMFS 0x00000004
#define KEXEC_FILE_DEBUG 0x00000008
#define KEXEC_ARCH_DEFAULT (0 << 16)
#define KEXEC_ARCH_386 (3 << 16)
#define KEXEC_ARCH_68K (4 << 16)
#define KEXEC_ARCH_PARISC (15 << 16)
#define KEXEC_ARCH_X86_64 (62 << 16)
#define KEXEC_ARCH_PPC (20 << 16)
#define KEXEC_ARCH_PPC64 (21 << 16)
#define KEXEC_ARCH_IA_64 (50 << 16)
#define KEXEC_ARCH_ARM (40 << 16)
#define KEXEC_ARCH_S390 (22 << 16)
#define KEXEC_ARCH_SH (42 << 16)
#define KEXEC_ARCH_MIPS_LE (10 << 16)
#define KEXEC_ARCH_MIPS (8 << 16)
#define KEXEC_ARCH_AARCH64 (183 << 16)
#define KEXEC_ARCH_RISCV (243 << 16)
#define KEXEC_ARCH_LOONGARCH (258 << 16)
#define KEXEC_SEGMENT_MAX 16
struct kexec_segment {
  const void * buf;
  __kernel_size_t bufsz;
  const void * mem;
  __kernel_size_t memsz;
};
#endif
