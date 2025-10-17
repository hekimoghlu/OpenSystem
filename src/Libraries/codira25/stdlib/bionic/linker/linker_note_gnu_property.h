/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#pragma once

#include <elf.h>
#include <link.h>

#include "linker_soinfo.h"

// The Elf* structures below are derived from the document
// Linux Extensions to gABI (https://github.com/hjl-tools/linux-abi/wiki).
// Essentially, these types would be defined in <elf.h>, but this is not
// the case at the moment.

struct Elf32_Prop {
  Elf32_Word pr_type;
  Elf32_Word pr_datasz;
  char pr_data[0];
};

// On 32-bit machines this should be 4-byte aligned.
struct Elf32_NhdrGNUProperty {
  Elf32_Nhdr nhdr;
  char n_name[4];
  char n_desc[0];
};

struct Elf64_Prop {
  Elf64_Word pr_type;
  Elf64_Word pr_datasz;
  char pr_data[0];
};

// On 64-bit machines this should be 8-byte aligned.
struct Elf64_NhdrGNUProperty {
  Elf64_Nhdr nhdr;
  char n_name[4];
  char n_desc[0];
};

struct ElfProgramProperty {
#if defined(__aarch64__)
  bool bti_compatible = false;
#endif
};

// Representation of the .note.gnu.property section found in the segment
// with p_type = PT_GNU_PROPERTY.
class GnuPropertySection {
 public:
  GnuPropertySection(){};
  explicit GnuPropertySection(const soinfo* si);
  GnuPropertySection(const ElfW(Phdr)* phdr, size_t phdr_count, const ElfW(Addr) load_bias,
                     const char* name);

#if defined(__aarch64__)
  bool IsBTICompatible() const;
#endif

 private:
  const ElfW(NhdrGNUProperty)* FindSegment(const ElfW(Phdr)* phdr, size_t phdr_count,
                                           const ElfW(Addr) load_bias, const char* name) const;
  bool SanityCheck(const ElfW(NhdrGNUProperty)* note_nhdr, const char* name) const;
  bool Parse(const ElfW(NhdrGNUProperty)* note_nhdr, const char* name);

  ElfProgramProperty properties_ __unused;
};
