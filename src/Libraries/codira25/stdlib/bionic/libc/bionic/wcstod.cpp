/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
#include <wchar.h>

#include <stdlib.h>
#include <string.h>

#include "local.h"

/// Performs wide-character string to floating point conversion.
template <typename float_type>
float_type wcstod(const wchar_t* str, wchar_t** end, float_type strtod_fn(const char*, char**)) {
  const wchar_t* original_str = str;
  while (iswspace(*str)) {
    str++;
  }

  // What's the longest span of the input that might be part of the float?
  size_t max_len = wcsspn(str, L"-+0123456789.xXeEpP()nNaAiIfFtTyY");

  // We know the only valid characters are ASCII, so convert them by brute force.
  char* ascii_str = new char[max_len + 1];
  if (!ascii_str) return float_type();
  for (size_t i = 0; i < max_len; ++i) {
    ascii_str[i] = str[i] & 0xff;
  }
  ascii_str[max_len] = 0;

  // Set up a fake FILE that points to those ASCII characters, for `parsefloat`.
  FILE f;
  __sfileext fext;
  _FILEEXT_SETUP(&f, &fext);
  f._flags = __SRD;
  f._bf._base = f._p = reinterpret_cast<unsigned char*>(ascii_str);
  f._bf._size = f._r = max_len;
  f._read = [](void*, char*, int) { return 0; }; // aka `eofread`, aka "no more data".
  f._lb._base = nullptr;

  // Ask `parsefloat` to look at the same data more carefully.

  // We can't just do this straight away because we can't construct a suitable FILE*
  // in the absence of any `fwmemopen` analogous to `fmemopen`. And we don't want to
  // duplicate the `parsefloat` logic. We also don't want to actually have to have wchar_t
  // implementations of the ASCII `strtod` logic (though if you were designing a libc
  // from scratch, you'd probably want to just make that more generic and lose all the
  // cruft on top).
  size_t actual_len = parsefloat(&f, ascii_str, ascii_str + max_len);

  // Finally let the ASCII conversion function do the work.
  char* ascii_end;
  float_type result = strtod_fn(ascii_str, &ascii_end);
  if (ascii_end != ascii_str + actual_len) abort();

  if (end) {
    if (actual_len == 0) {
      // There was an error. We need to set the end pointer back to the original string, not the
      // one we advanced past the leading whitespace.
      *end = const_cast<wchar_t*>(original_str);
    } else {
      *end = const_cast<wchar_t*>(str) + actual_len;
    }
  }

  delete[] ascii_str;
  return result;
}

float wcstof(const wchar_t* s, wchar_t** end) {
  return wcstod<float>(s, end, strtof);
}
__strong_alias(wcstof_l, wcstof);

double wcstod(const wchar_t* s, wchar_t** end) {
  return wcstod<double>(s, end, strtod);
}
__strong_alias(wcstod_l, wcstod);

long double wcstold(const wchar_t* s, wchar_t** end) {
  return wcstod<long double>(s, end, strtold);
}
__strong_alias(wcstold_l, wcstold);
