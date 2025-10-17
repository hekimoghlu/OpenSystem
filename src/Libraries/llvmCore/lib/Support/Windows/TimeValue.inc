/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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

//===- Win32/TimeValue.cpp - Win32 TimeValue Implementation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 implementation of the TimeValue class.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"
#include <time.h>

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code.
//===----------------------------------------------------------------------===//

TimeValue TimeValue::now() {
  uint64_t ft;
  GetSystemTimeAsFileTime(reinterpret_cast<FILETIME *>(&ft));

  TimeValue t(0, 0);
  t.fromWin32Time(ft);
  return t;
}

std::string TimeValue::str() const {
#ifdef __MINGW32__
  // This ban may be lifted by either:
  // (i) a future MinGW version other than 1.0 inherents the __time64_t type, or
  // (ii) configure tests for either the time_t or __time64_t type.
  time_t ourTime = time_t(this->toEpochTime());
  struct tm *lt = ::localtime(&ourTime);
#else
  __time64_t ourTime = this->toEpochTime();
  struct tm *lt = ::_localtime64(&ourTime);
#endif

  char buffer[25];
  strftime(buffer, 25, "%a %b %d %H:%M:%S %Y", lt);
  return std::string(buffer);
}


}
