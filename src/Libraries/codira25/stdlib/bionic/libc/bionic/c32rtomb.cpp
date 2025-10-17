/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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

size_t c32rtomb(char* s, char32_t c32, mbstate_t* ps) {
  static mbstate_t __private_state;
  mbstate_t* state = (ps == nullptr) ? &__private_state : ps;

  if (s == nullptr) {
    // Equivalent to c32rtomb(buf, U'\0', ps).
    return mbstate_reset_and_return(1, state);
  }

  // POSIX states that if char32_t is a null wide character, a null byte shall
  // be stored, preceded by any shift sequence needed to restore the initial
  // shift state. Since shift states are not supported, only the null byte is
  // stored.
  if (c32 == U'\0') {
    *s = '\0';
    return mbstate_reset_and_return(1, state);
  }

  if (!mbstate_is_initial(state)) {
    return mbstate_reset_and_return_illegal(EILSEQ, state);
  }

  if ((c32 & ~0x7f) == 0) {
    // Fast path for plain ASCII characters.
    *s = c32;
    return 1;
  }

  // Determine the number of octets needed to represent this character.
  // We always output the shortest sequence possible. Also specify the
  // first few bits of the first octet, which contains the information
  // about the sequence length.
  uint8_t lead;
  size_t length;
  // We already handled the 1-byte case above, so we go straight to 2-bytes...
  if ((c32 & ~0x7ff) == 0) {
    lead = 0xc0;
    length = 2;
  } else if ((c32 & ~0xffff) == 0) {
    lead = 0xe0;
    length = 3;
  } else if ((c32 & ~0x1fffff) == 0) {
    lead = 0xf0;
    length = 4;
  } else {
    errno = EILSEQ;
    return BIONIC_MULTIBYTE_RESULT_ILLEGAL_SEQUENCE;
  }

  // Output the octets representing the character in chunks
  // of 6 bits, least significant last. The first octet is
  // a special case because it contains the sequence length
  // information.
  for (size_t i = length - 1; i > 0; i--) {
    s[i] = (c32 & 0x3f) | 0x80;
    c32 >>= 6;
  }
  *s = (c32 & 0xff) | lead;

  return length;
}
