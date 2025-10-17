/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

//===--- Win32.cpp - Win32 utility functions --------------------*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//
//
// Utility functions that are specific to the Windows port.
//
//===----------------------------------------------------------------------===//

#include "language/Runtime/Debug.h"
#include "language/Runtime/Win32.h"

#ifdef _WIN32

#include <windows.h>

char *
_language_win32_copyUTF8FromWide(const wchar_t *str) {
  char *result = nullptr;
  int len = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                  str, -1,
                                  nullptr, 0,
                                  nullptr, nullptr);
  if (len <= 0)
    return nullptr;

  result = reinterpret_cast<char *>(std::malloc(len));
  if (!result)
    return nullptr;

  len = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                              str, -1,
                              result, len,
                              nullptr, nullptr);

  if (len)
    return result;

  free(result);
  return nullptr;
}

wchar_t *
_language_win32_copyWideFromUTF8(const char *str) {
  wchar_t *result = nullptr;
  int len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                  str, -1,
                                  nullptr, 0);
  if (len <= 0)
    return nullptr;

  result = reinterpret_cast<wchar_t *>(std::malloc(len * sizeof(wchar_t)));
  if (!result)
    return nullptr;

  len = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                              str, -1,
                              result, len);

  if (len)
    return result;

  free(result);
  return nullptr;
}

#endif // defined(_WIN32)
