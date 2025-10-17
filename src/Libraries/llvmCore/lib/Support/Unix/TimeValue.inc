/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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

//===- Unix/TimeValue.cpp - Unix TimeValue Implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific portion of the TimeValue class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"

namespace llvm {
  using namespace sys;

std::string TimeValue::str() const {
  char buffer[32];

  time_t ourTime = time_t(this->toEpochTime());
#ifdef __hpux
// note that the following line needs -D_REENTRANT on HP-UX to be picked up
  asctime_r(localtime(&ourTime), buffer);
#else
  ::asctime_r(::localtime(&ourTime), buffer);
#endif

  std::string result(buffer);
  return result.substr(0,24);
}

TimeValue TimeValue::now() {
  struct timeval the_time;
  timerclear(&the_time);
  if (0 != ::gettimeofday(&the_time,0)) {
    // This is *really* unlikely to occur because the only gettimeofday
    // errors concern the timezone parameter which we're passing in as 0.
    // In the unlikely case it does happen, just return MinTime, no error
    // message needed.
    return MinTime;
  }

  return TimeValue(
    static_cast<TimeValue::SecondsType>( the_time.tv_sec + PosixZeroTime.seconds_ ),
    static_cast<TimeValue::NanoSecondsType>( the_time.tv_usec *
      NANOSECONDS_PER_MICROSECOND ) );
}

}
