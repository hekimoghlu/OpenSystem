/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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

#include <link.h>
#include <stdint.h>
#include <stdlib.h>

#include <utility>
#include <vector>

#include "linker_common_types.h"
#include "linker_globals.h"
#include "linker_soinfo.h"

static constexpr ElfW(Versym) kVersymHiddenBit = 0x8000;

enum RelocationKind {
  kRelocAbsolute = 0,
  kRelocRelative,
  kRelocSymbol,
  kRelocSymbolCached,
  kRelocMax
};

void count_relocation(RelocationKind kind);

template <bool Enabled> void count_relocation_if(RelocationKind kind) {
  if (Enabled) count_relocation(kind);
}

void print_linker_stats();

inline bool is_symbol_global_and_defined(const soinfo* si, const ElfW(Sym)* s) {
  if (__predict_true(ELF_ST_BIND(s->st_info) == STB_GLOBAL ||
                     ELF_ST_BIND(s->st_info) == STB_WEAK)) {
    return s->st_shndx != SHN_UNDEF;
  } else if (__predict_false(ELF_ST_BIND(s->st_info) != STB_LOCAL)) {
    DL_WARN("Warning: unexpected ST_BIND value: %d for \"%s\" in \"%s\" (ignoring)",
            ELF_ST_BIND(s->st_info), si->get_string(s->st_name), si->get_realpath());
  }
  return false;
}
