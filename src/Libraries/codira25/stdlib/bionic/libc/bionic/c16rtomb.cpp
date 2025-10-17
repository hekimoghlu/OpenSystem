/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#include <errno.h>
#include <uchar.h>
#include <wchar.h>

#include "private/bionic_mbstate.h"

static inline constexpr bool is_high_surrogate(char16_t c16) {
  return c16 >= 0xd800 && c16 < 0xdc00;
}

static inline constexpr bool is_low_surrogate(char16_t c16) {
  return c16 >= 0xdc00 && c16 < 0xe000;
}

size_t c16rtomb(char* s, char16_t c16, mbstate_t* ps) {
  static mbstate_t __private_state;
  mbstate_t* state = (ps == nullptr) ? &__private_state : ps;
  if (mbstate_is_initial(state)) {
    if (is_high_surrogate(c16)) {
      char32_t c32 = (c16 & ~0xd800) << 10;
      mbstate_set_byte(state, 3, (c32 & 0xff0000) >> 16);
      mbstate_set_byte(state, 2, (c32 & 0x00ff00) >> 8);
      return 0;
    } else if (is_low_surrogate(c16)) {
      return mbstate_reset_and_return_illegal(EINVAL, state);
    } else {
      return c32rtomb(s, static_cast<char32_t>(c16), state);
    }
  } else {
    if (!is_low_surrogate(c16)) {
      return mbstate_reset_and_return_illegal(EINVAL, state);
    }

    char32_t c32 = ((mbstate_get_byte(state, 3) << 16) |
                    (mbstate_get_byte(state, 2) << 8) |
                    (c16 & ~0xdc00)) + 0x10000;
    return mbstate_reset_and_return(c32rtomb(s, c32, nullptr), state);
  }
}
