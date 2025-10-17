/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
// -std=gnu++XX automatically defines _GNU_SOURCE, which then means that <string.h>
// gives us the GNU variant, which is not what we're defining here.
#undef _GNU_SOURCE

#include <string.h>

#include <errno.h>
#include <limits.h>

#include <async_safe/log.h>

#include "private/ErrnoRestorer.h"

#include <string.h>

#include "bionic/pthread_internal.h"

static const char* __sys_error_descriptions[] = {
#define __BIONIC_ERRDEF(error_number, error_description) [error_number] = error_description,
#include "private/bionic_errdefs.h"
};

static const char* __sys_error_names[] = {
#define __BIONIC_ERRDEF(error_number, error_description) [error_number] = #error_number,
#include "private/bionic_errdefs.h"
};

extern "C" const char* strerrorname_np(int error_number) {
  if (error_number < 0 || error_number >= static_cast<int>(arraysize(__sys_error_names))) {
    return nullptr;
  }
  return __sys_error_names[error_number];
}

static inline const char* __strerror_lookup(int error_number) {
  if (error_number < 0 || error_number >= static_cast<int>(arraysize(__sys_error_descriptions))) {
    return nullptr;
  }
  return __sys_error_descriptions[error_number];
}

int strerror_r(int error_number, char* buf, size_t buf_len) {
  ErrnoRestorer errno_restorer;
  size_t length;

  const char* error_name = __strerror_lookup(error_number);
  if (error_name != nullptr) {
    length = strlcpy(buf, error_name, buf_len);
  } else {
    length = async_safe_format_buffer(buf, buf_len, "Unknown error %d", error_number);
  }
  if (length >= buf_len) {
    return ERANGE;
  }

  return 0;
}

extern "C" char* __gnu_strerror_r(int error_number, char* buf, size_t buf_len) {
  ErrnoRestorer errno_restorer; // The glibc strerror_r doesn't set errno if it truncates...
  strerror_r(error_number, buf, buf_len);
  return buf; // ...and just returns whatever fit.
}

char* strerror(int error_number) {
  // Just return the original constant in the easy cases.
  char* result = const_cast<char*>(__strerror_lookup(error_number));
  if (result != nullptr) {
    return result;
  }

  bionic_tls& tls = __get_bionic_tls();
  result = tls.strerror_buf;
  strerror_r(error_number, result, sizeof(tls.strerror_buf));
  return result;
}
__strong_alias(strerror_l, strerror);
