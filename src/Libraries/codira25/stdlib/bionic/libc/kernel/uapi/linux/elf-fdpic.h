/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
#ifndef _UAPI_LINUX_ELF_FDPIC_H
#define _UAPI_LINUX_ELF_FDPIC_H
#include <linux/elf.h>
#define PT_GNU_STACK (PT_LOOS + 0x474e551)
struct elf32_fdpic_loadseg {
  Elf32_Addr addr;
  Elf32_Addr p_vaddr;
  Elf32_Word p_memsz;
};
struct elf32_fdpic_loadmap {
  Elf32_Half version;
  Elf32_Half nsegs;
  struct elf32_fdpic_loadseg segs[];
};
#define ELF32_FDPIC_LOADMAP_VERSION 0x0000
struct elf64_fdpic_loadseg {
  Elf64_Addr addr;
  Elf64_Addr p_vaddr;
  Elf64_Word p_memsz;
};
struct elf64_fdpic_loadmap {
  Elf64_Half version;
  Elf64_Half nsegs;
  struct elf64_fdpic_loadseg segs[];
};
#define ELF64_FDPIC_LOADMAP_VERSION 0x0000
#endif
