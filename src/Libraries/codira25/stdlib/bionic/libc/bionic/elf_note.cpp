/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
#include "platform/bionic/macros.h"
#include "private/elf_note.h"

#include <string.h>

bool __get_elf_note(unsigned note_type, const char* note_name, const ElfW(Addr) note_addr,
                    const ElfW(Phdr)* phdr_note, const ElfW(Nhdr)** note_hdr,
                    const char** note_desc) {
  if (phdr_note->p_type != PT_NOTE || !note_name || !note_addr) {
    return false;
  }

  size_t note_name_len = strlen(note_name) + 1;

  ElfW(Addr) p = note_addr;
  ElfW(Addr) note_end = p + phdr_note->p_memsz;
  while (p < note_end) {
    // Parse the note and check it's structurally valid.
    const ElfW(Nhdr)* note = reinterpret_cast<const ElfW(Nhdr)*>(p);
    if (__builtin_add_overflow(p, sizeof(ElfW(Nhdr)), &p) || p >= note_end) {
      return false;
    }
    const char* name = reinterpret_cast<const char*>(p);
    if (__builtin_add_overflow(p, __builtin_align_up(note->n_namesz, 4), &p)) {
      return false;
    }
    const char* desc = reinterpret_cast<const char*>(p);
    if (__builtin_add_overflow(p, __builtin_align_up(note->n_descsz, 4), &p)) {
      return false;
    }
    if (p > note_end) {
      return false;
    }

    // Is this the note we're looking for?
    if (note->n_type == note_type &&
        note->n_namesz == note_name_len &&
        strncmp(note_name, name, note_name_len) == 0) {
      *note_hdr = note;
      *note_desc = desc;
      return true;
    }
  }
  return false;
}

bool __find_elf_note(unsigned int note_type, const char* note_name, const ElfW(Phdr)* phdr_start,
                     size_t phdr_ct, const ElfW(Nhdr)** note_hdr, const char** note_desc,
                     const ElfW(Addr) load_bias) {
  for (size_t i = 0; i < phdr_ct; ++i) {
    const ElfW(Phdr)* phdr = &phdr_start[i];

    ElfW(Addr) note_addr = load_bias + phdr->p_vaddr;
    if (__get_elf_note(note_type, note_name, note_addr, phdr, note_hdr, note_desc)) {
      return true;
    }
  }

  return false;
}
