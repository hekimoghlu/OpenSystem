/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#include "modules/audio_coding/neteq/tools/rtp_generator.h"

namespace webrtc {
namespace test {

uint32_t RtpGenerator::GetRtpHeader(uint8_t payload_type,
                                    size_t payload_length_samples,
                                    RTPHeader* rtp_header) {
  RTC_DCHECK(rtp_header);
  if (!rtp_header) {
    return 0;
  }
  rtp_header->sequenceNumber = seq_number_++;
  rtp_header->timestamp = timestamp_;
  timestamp_ += static_cast<uint32_t>(payload_length_samples);
  rtp_header->payloadType = payload_type;
  rtp_header->markerBit = false;
  rtp_header->ssrc = ssrc_;
  rtp_header->numCSRCs = 0;

  uint32_t this_send_time = next_send_time_ms_;
  RTC_DCHECK_GT(samples_per_ms_, 0);
  next_send_time_ms_ +=
      ((1.0 + drift_factor_) * payload_length_samples) / samples_per_ms_;
  return this_send_time;
}

void RtpGenerator::set_drift_factor(double factor) {
  if (factor > -1.0) {
    drift_factor_ = factor;
  }
}

uint32_t TimestampJumpRtpGenerator::GetRtpHeader(uint8_t payload_type,
                                                 size_t payload_length_samples,
                                                 RTPHeader* rtp_header) {
  uint32_t ret = RtpGenerator::GetRtpHeader(payload_type,
                                            payload_length_samples, rtp_header);
  if (timestamp_ - static_cast<uint32_t>(payload_length_samples) <=
          jump_from_timestamp_ &&
      timestamp_ > jump_from_timestamp_) {
    // We just moved across the `jump_from_timestamp_` timestamp. Do the jump.
    timestamp_ = jump_to_timestamp_;
  }
  return ret;
}

}  // namespace test
}  // namespace webrtc
