/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
#ifndef API_METRONOME_METRONOME_H_
#define API_METRONOME_METRONOME_H_

#include "absl/functional/any_invocable.h"
#include "api/units/time_delta.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// The Metronome posts OnTick() calls requested with RequestCallOnNextTick.
// The API is designed to be fully used from a single task queue. Scheduled
// callbacks are executed on the same sequence as they were requested on. There
// are no features implemented for cancellation. When that's needed, use e.g.
// ScopedTaskSafety from the client.
//
// The metronome concept is still under experimentation, and may not be availble
// in all platforms or applications. See https://crbug.com/1253787 for more
// details.
//
// Metronome implementations must be thread-compatible.
class RTC_EXPORT Metronome {
 public:
  virtual ~Metronome() = default;

  // Requests a call to `callback` on the next tick. Scheduled callbacks are
  // executed on the same sequence as they were requested on. There are no
  // features for cancellation. When that's needed, use e.g. ScopedTaskSafety
  // from the client.
  virtual void RequestCallOnNextTick(
      absl::AnyInvocable<void() &&> /* callback */) {}

  // Returns the current tick period of the metronome.
  virtual TimeDelta TickPeriod() const = 0;
};

}  // namespace webrtc

#endif  // API_METRONOME_METRONOME_H_
