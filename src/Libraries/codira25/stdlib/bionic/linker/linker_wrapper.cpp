/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#include "private/KernelArgumentBlock.h"

// The offset from the linker's original program header load addresses to
// the load addresses when embedded into a binary.  Set by the extract_linker
// tool.
extern const char __dlwrap_linker_offset;

// The real entry point of the binary to use after linker bootstrapping.
__LIBC_HIDDEN__ extern "C" void _start();

/* Find the load bias and base address of an executable or shared object loaded
 * by the kernel. The ELF file's PHDR table must have a PT_PHDR entry.
 *
 * A VDSO doesn't have a PT_PHDR entry in its PHDR table.
 */
static void get_elf_base_from_phdr(const ElfW(Phdr)* phdr_table, size_t phdr_count,
                                   ElfW(Addr)* base, ElfW(Addr)* load_bias) {
  for (size_t i = 0; i < phdr_count; ++i) {
    if (phdr_table[i].p_type == PT_PHDR) {
      *load_bias = reinterpret_cast<ElfW(Addr)>(phdr_table) - phdr_table[i].p_vaddr;
      *base = reinterpret_cast<ElfW(Addr)>(phdr_table) - phdr_table[i].p_offset;
      return;
    }
  }
}

/*
 * This is the entry point for the linker wrapper, which finds
 * the real linker, then bootstraps into it.
 */
extern "C" ElfW(Addr) __linker_init(void* raw_args) {
  KernelArgumentBlock args(raw_args);

  ElfW(Addr) base_addr = 0;
  ElfW(Addr) load_bias = 0;
  get_elf_base_from_phdr(
    reinterpret_cast<ElfW(Phdr)*>(args.getauxval(AT_PHDR)), args.getauxval(AT_PHNUM),
    &base_addr, &load_bias);

  ElfW(Addr) linker_addr = base_addr + reinterpret_cast<uintptr_t>(&__dlwrap_linker_offset);
  ElfW(Addr) linker_entry_offset = reinterpret_cast<ElfW(Ehdr)*>(linker_addr)->e_entry;

  for (ElfW(auxv_t)* v = args.auxv; v->a_type != AT_NULL; ++v) {
    if (v->a_type == AT_BASE) {
      // Set AT_BASE to the embedded linker
      v->a_un.a_val = linker_addr;
    }
    if (v->a_type == AT_ENTRY) {
      // Set AT_ENTRY to the proper entry point
      v->a_un.a_val = reinterpret_cast<ElfW(Addr)>(&_start);
    }
  }

  // Return address of linker entry point
  return linker_addr + linker_entry_offset;
}
