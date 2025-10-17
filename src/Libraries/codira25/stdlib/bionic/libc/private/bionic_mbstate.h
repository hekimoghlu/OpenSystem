/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#ifndef _BIONIC_MBSTATE_H
#define _BIONIC_MBSTATE_H

#include <errno.h>
#include <wchar.h>

__BEGIN_DECLS

#define __MB_IS_ERR(rv)                              \
  (rv == BIONIC_MULTIBYTE_RESULT_ILLEGAL_SEQUENCE || \
   rv == BIONIC_MULTIBYTE_RESULT_INCOMPLETE_SEQUENCE)

static inline __nodiscard bool mbstate_is_initial(const mbstate_t* ps) {
  return *(reinterpret_cast<const uint32_t*>(ps->__seq)) == 0;
}

static inline __nodiscard size_t mbstate_bytes_so_far(const mbstate_t* ps) {
  return
      (ps->__seq[2] != 0) ? 3 :
      (ps->__seq[1] != 0) ? 2 :
      (ps->__seq[0] != 0) ? 1 : 0;
}

static inline void mbstate_set_byte(mbstate_t* ps, int i, char byte) {
  ps->__seq[i] = static_cast<uint8_t>(byte);
}

static inline __nodiscard uint8_t mbstate_get_byte(const mbstate_t* ps, int n) {
  return ps->__seq[n];
}

static inline void mbstate_reset(mbstate_t* ps) {
  *(reinterpret_cast<uint32_t*>(ps->__seq)) = 0;
}

static inline __nodiscard size_t mbstate_reset_and_return_illegal(int _errno, mbstate_t* ps) {
  errno = _errno;
  mbstate_reset(ps);
  return BIONIC_MULTIBYTE_RESULT_ILLEGAL_SEQUENCE;
}

static inline __nodiscard size_t mbstate_reset_and_return(size_t _return, mbstate_t* ps) {
  mbstate_reset(ps);
  return _return;
}

__END_DECLS

#endif // _BIONIC_MBSTATE_H
