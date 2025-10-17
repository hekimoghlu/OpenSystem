/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

#include <string.h>

#include "linker.h"
#include "linker_sleb128.h"

const size_t RELOCATION_GROUPED_BY_INFO_FLAG = 1;
const size_t RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG = 2;
const size_t RELOCATION_GROUPED_BY_ADDEND_FLAG = 4;
const size_t RELOCATION_GROUP_HAS_ADDEND_FLAG = 8;

#if defined(USE_RELA)
typedef ElfW(Rela) rel_t;
#else
typedef ElfW(Rel) rel_t;
#endif

template <typename F>
inline bool for_all_packed_relocs(sleb128_decoder decoder, F&& callback) {
  const size_t num_relocs = decoder.pop_front();

  rel_t reloc = {
    .r_offset = decoder.pop_front(),
  };

  for (size_t idx = 0; idx < num_relocs; ) {
    const size_t group_size = decoder.pop_front();
    const size_t group_flags = decoder.pop_front();

    size_t group_r_offset_delta = 0;

    if (group_flags & RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG) {
      group_r_offset_delta = decoder.pop_front();
    }
    if (group_flags & RELOCATION_GROUPED_BY_INFO_FLAG) {
      reloc.r_info = decoder.pop_front();
    }

#if defined(USE_RELA)
    const size_t group_flags_reloc = group_flags & (RELOCATION_GROUP_HAS_ADDEND_FLAG |
                                                    RELOCATION_GROUPED_BY_ADDEND_FLAG);
    if (group_flags_reloc == RELOCATION_GROUP_HAS_ADDEND_FLAG) {
      // Each relocation has an addend. This is the default situation with lld's current encoder.
    } else if (group_flags_reloc == (RELOCATION_GROUP_HAS_ADDEND_FLAG |
                                     RELOCATION_GROUPED_BY_ADDEND_FLAG)) {
      reloc.r_addend += decoder.pop_front();
    } else {
      reloc.r_addend = 0;
    }
#else
    if (__predict_false(group_flags & RELOCATION_GROUP_HAS_ADDEND_FLAG)) {
      // This platform does not support rela, and yet we have it encoded in android_rel section.
      async_safe_fatal("unexpected r_addend in android.rel section");
    }
#endif

    for (size_t i = 0; i < group_size; ++i) {
      if (group_flags & RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG) {
        reloc.r_offset += group_r_offset_delta;
      } else {
        reloc.r_offset += decoder.pop_front();
      }
      if ((group_flags & RELOCATION_GROUPED_BY_INFO_FLAG) == 0) {
        reloc.r_info = decoder.pop_front();
      }
#if defined(USE_RELA)
      if (group_flags_reloc == RELOCATION_GROUP_HAS_ADDEND_FLAG) {
        reloc.r_addend += decoder.pop_front();
      }
#endif
      if (!callback(reloc)) {
        return false;
      }
    }

    idx += group_size;
  }

  return true;
}
