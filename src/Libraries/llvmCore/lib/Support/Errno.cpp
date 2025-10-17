/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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

//===- Errno.cpp - errno support --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the errno wrappers.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Errno.h"
#include "llvm/Config/config.h"     // Get autoconf configuration settings
#include "llvm/Support/raw_ostream.h"

#if HAVE_STRING_H
#include <string.h>

#if HAVE_ERRNO_H
#include <errno.h>
#endif

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

namespace llvm {
namespace sys {

#if HAVE_ERRNO_H
std::string StrError() {
  return StrError(errno);
}
#endif  // HAVE_ERRNO_H

std::string StrError(int errnum) {
  const int MaxErrStrLen = 2000;
  char buffer[MaxErrStrLen];
  buffer[0] = '\0';
  std::string str;
#ifdef HAVE_STRERROR_R
  // strerror_r is thread-safe.
  if (errnum)
# if defined(__GLIBC__) && defined(_GNU_SOURCE)
    // glibc defines its own incompatible version of strerror_r
    // which may not use the buffer supplied.
    str = strerror_r(errnum,buffer,MaxErrStrLen-1);
# else
    strerror_r(errnum,buffer,MaxErrStrLen-1);
    str = buffer;
# endif
#elif HAVE_DECL_STRERROR_S // "Windows Secure API"
    if (errnum)
      strerror_s(buffer, MaxErrStrLen - 1, errnum);
#elif defined(HAVE_STRERROR)
  // Copy the thread un-safe result of strerror into
  // the buffer as fast as possible to minimize impact
  // of collision of strerror in multiple threads.
  if (errnum)
    str = strerror(errnum);
#else
  // Strange that this system doesn't even have strerror
  // but, oh well, just use a generic message
  raw_string_ostream stream(str);
  stream << "Error #" << errnum;
  stream.flush();
#endif
  return str;
}

}  // namespace sys
}  // namespace llvm

#endif  // HAVE_STRING_H
