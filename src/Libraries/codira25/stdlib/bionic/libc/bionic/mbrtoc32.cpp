/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#include <sys/param.h>
#include <uchar.h>
#include <wchar.h>

#include "private/bionic_mbstate.h"

size_t mbrtoc32(char32_t* pc32, const char* s, size_t n, mbstate_t* ps) {
  static mbstate_t __private_state;
  mbstate_t* state = (ps == nullptr) ? &__private_state : ps;

  // We should never get to a state which has all 4 bytes of the sequence set.
  // Full state verification is done when decoding the sequence (after we have
  // all the bytes).
  if (mbstate_get_byte(state, 3) != 0) {
    return mbstate_reset_and_return_illegal(EINVAL, state);
  }

  if (s == nullptr) {
    s = "";
    n = 1;
    pc32 = nullptr;
  }

  if (n == 0) {
    // C23 7.30.1 (for each `mbrtoc*` function) says:
    //
    // Returns:
    //
    //     0 if the next n or fewer bytes complete the multibyte character that
    //     corresponds to the null wide character (which is the value stored).
    //
    //     (size_t)(-2) if the next n bytes contribute to an incomplete (but
    //     potentially valid) multibyte character, and all n bytes have been
    //     processed (no value is stored).
    //
    // Bionic historically interpreted the behavior when n is 0 to be the next 0
    // bytes decoding to the null. That's a pretty bad interpretation, and both
    // glibc and musl return -2 for that case.
    return BIONIC_MULTIBYTE_RESULT_INCOMPLETE_SEQUENCE;
  }

  uint8_t ch;
  if (mbstate_is_initial(state) && (((ch = static_cast<uint8_t>(*s)) & ~0x7f) == 0)) {
    // Fast path for plain ASCII characters.
    if (pc32 != nullptr) {
      *pc32 = ch;
    }
    return (ch != '\0' ? 1 : 0);
  }

  // Determine the number of octets that make up this character
  // from the first octet, and a mask that extracts the
  // interesting bits of the first octet. We already know
  // the character is at least two bytes long.
  size_t length;
  int mask;

  // We also specify a lower bound for the character code to
  // detect redundant, non-"shortest form" encodings. For
  // example, the sequence C0 80 is _not_ a legal representation
  // of the null character. This enforces a 1-to-1 mapping
  // between character codes and their multibyte representations.
  char32_t lower_bound;

  // The first byte in the state (if any) tells the length.
  size_t bytes_so_far = mbstate_bytes_so_far(state);
  ch = bytes_so_far > 0 ? mbstate_get_byte(state, 0) : static_cast<uint8_t>(*s);
  // We already handled the 1-byte case above, so we go straight to 2-bytes...
  if ((ch & 0xe0) == 0xc0) {
    mask = 0x1f;
    length = 2;
    lower_bound = 0x80;
  } else if ((ch & 0xf0) == 0xe0) {
    mask = 0x0f;
    length = 3;
    lower_bound = 0x800;
  } else if ((ch & 0xf8) == 0xf0) {
    mask = 0x07;
    length = 4;
    lower_bound = 0x10000;
  } else {
    // Malformed input; input is not UTF-8. See RFC 3629.
    return mbstate_reset_and_return_illegal(EILSEQ, state);
  }

  // Fill in the state.
  size_t bytes_wanted = length - bytes_so_far;
  size_t i;
  for (i = 0; i < MIN(bytes_wanted, n); i++) {
    if (!mbstate_is_initial(state) && ((*s & 0xc0) != 0x80)) {
      // Malformed input; bad characters in the middle of a character.
      return mbstate_reset_and_return_illegal(EILSEQ, state);
    }
    mbstate_set_byte(state, bytes_so_far + i, *s++);
  }
  if (i < bytes_wanted) {
    return BIONIC_MULTIBYTE_RESULT_INCOMPLETE_SEQUENCE;
  }

  // Decode the octet sequence representing the character in chunks
  // of 6 bits, most significant first.
  char32_t c32 = mbstate_get_byte(state, 0) & mask;
  for (i = 1; i < length; i++) {
    c32 <<= 6;
    c32 |= mbstate_get_byte(state, i) & 0x3f;
  }

  if (c32 < lower_bound) {
    // Malformed input; redundant encoding.
    return mbstate_reset_and_return_illegal(EILSEQ, state);
  }
  if ((c32 >= 0xd800 && c32 <= 0xdfff) || (c32 > 0x10ffff)) {
    // Malformed input; invalid code points.
    return mbstate_reset_and_return_illegal(EILSEQ, state);
  }
  if (pc32 != nullptr) {
    *pc32 = c32;
  }
  return mbstate_reset_and_return(c32 == U'\0' ? 0 : bytes_wanted, state);
}
