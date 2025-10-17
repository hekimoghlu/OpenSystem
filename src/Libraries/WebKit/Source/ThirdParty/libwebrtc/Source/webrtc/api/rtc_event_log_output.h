/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
#ifndef API_RTC_EVENT_LOG_OUTPUT_H_
#define API_RTC_EVENT_LOG_OUTPUT_H_


#include "absl/strings/string_view.h"

namespace webrtc {

// NOTE: This class is still under development and may change without notice.
class RtcEventLogOutput {
 public:
  virtual ~RtcEventLogOutput() = default;

  // An output normally starts out active, though that might not always be
  // the case (e.g. failed to open a file for writing).
  // Once an output has become inactive (e.g. maximum file size reached), it can
  // never become active again.
  virtual bool IsActive() const = 0;

  // Write encoded events to an output. Returns true if the output was
  // successfully written in its entirety. Otherwise, no guarantee is given
  // about how much data was written, if any. The output sink becomes inactive
  // after the first time `false` is returned. Write() may not be called on
  // an inactive output sink.
  virtual bool Write(absl::string_view output) = 0;

  // Indicates that buffers should be written to disk if applicable.
  virtual void Flush() {}
};

}  // namespace webrtc

#endif  // API_RTC_EVENT_LOG_OUTPUT_H_
