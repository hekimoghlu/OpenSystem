/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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

//===--- Win32Extras.cpp - Windows support functions ------------*- C++ -*-===//
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
//  Defines some extra functions that aren't available in the OS or C library
//  on Windows.
//
//===----------------------------------------------------------------------===//

#ifdef _WIN32

#include <windows.h>

#include "modules/OS/Libc.h"

extern "C" ssize_t pread(int fd, void *buf, size_t nbyte, off_t offset) {
  HANDLE hFile = _get_osfhandle(fd);
  OVERLAPPED ovl = {0};
  DWORD dwBytesRead = 0;

  ovl.Offset = (DWORD)offset;
  ovl.OffsetHigh = (DWORD)(offset >> 32);

  if (!ReadFile(hFile, buf, (DWORD)count, &dwBytesRead, &ovl)) {
    errno = EIO;
    return -1;
  }

  return dwBytesRead;
}

extern "C" ssize_t pwrite(int fd, const void *buf, size_t nbyte, off_t offset) {
  HANDLE hFile = _get_osfhandle(fd);
  OVERLAPPED ovl = {0};
  DWORD dwBytesRead = 0;

  ovl.Offset = (DWORD)offset;
  ovl.OffsetHigh = (DWORD)(offset >> 32);

  if (!WriteFile(hFile, buf, (DWORD)count, &dwBytesRead, &ovl)) {
    errno = EIO;
    return -1;
  }

  return dwBytesRead;
}

#endif // _WIN32

