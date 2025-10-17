/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
#ifndef PC_JITTER_BUFFER_DELAY_H_
#define PC_JITTER_BUFFER_DELAY_H_

#include <stdint.h>

#include <optional>

#include "api/sequence_checker.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

// JitterBufferDelay converts delay from seconds to milliseconds for the
// underlying media channel. It also handles cases when user sets delay before
// the start of media_channel by caching its request.
class JitterBufferDelay {
 public:
  JitterBufferDelay() = default;

  void Set(std::optional<double> delay_seconds);
  int GetMs() const;

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker worker_thread_checker_{
      SequenceChecker::kDetached};
  std::optional<double> cached_delay_seconds_
      RTC_GUARDED_BY(&worker_thread_checker_);
};

}  // namespace webrtc

#endif  // PC_JITTER_BUFFER_DELAY_H_
