/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
// Syncable is used by RtpStreamsSynchronizer in VideoReceiveStreamInterface,
// and implemented by AudioReceiveStreamInterface.

#ifndef CALL_SYNCABLE_H_
#define CALL_SYNCABLE_H_

#include <stdint.h>

#include <optional>

namespace webrtc {

class Syncable {
 public:
  struct Info {
    int64_t latest_receive_time_ms = 0;
    uint32_t latest_received_capture_timestamp = 0;
    uint32_t capture_time_ntp_secs = 0;
    uint32_t capture_time_ntp_frac = 0;
    uint32_t capture_time_source_clock = 0;
    int current_delay_ms = 0;
  };

  virtual ~Syncable();

  virtual uint32_t id() const = 0;
  virtual std::optional<Info> GetInfo() const = 0;
  virtual bool GetPlayoutRtpTimestamp(uint32_t* rtp_timestamp,
                                      int64_t* time_ms) const = 0;
  virtual bool SetMinimumPlayoutDelay(int delay_ms) = 0;
  virtual void SetEstimatedPlayoutNtpTimestampMs(int64_t ntp_timestamp_ms,
                                                 int64_t time_ms) = 0;
};
}  // namespace webrtc

#endif  // CALL_SYNCABLE_H_
