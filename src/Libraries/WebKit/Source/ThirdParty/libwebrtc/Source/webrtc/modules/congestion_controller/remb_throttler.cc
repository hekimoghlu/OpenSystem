/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include "modules/congestion_controller/remb_throttler.h"

#include <algorithm>
#include <utility>

namespace webrtc {

namespace {
constexpr TimeDelta kRembSendInterval = TimeDelta::Millis(200);
}  // namespace

RembThrottler::RembThrottler(RembSender remb_sender, Clock* clock)
    : remb_sender_(std::move(remb_sender)),
      clock_(clock),
      last_remb_time_(Timestamp::MinusInfinity()),
      last_send_remb_bitrate_(DataRate::PlusInfinity()),
      max_remb_bitrate_(DataRate::PlusInfinity()) {}

void RembThrottler::OnReceiveBitrateChanged(const std::vector<uint32_t>& ssrcs,
                                            uint32_t bitrate_bps) {
  DataRate receive_bitrate = DataRate::BitsPerSec(bitrate_bps);
  Timestamp now = clock_->CurrentTime();
  {
    MutexLock lock(&mutex_);
    // % threshold for if we should send a new REMB asap.
    const int64_t kSendThresholdPercent = 103;
    if (receive_bitrate * kSendThresholdPercent / 100 >
            last_send_remb_bitrate_ &&
        now < last_remb_time_ + kRembSendInterval) {
      return;
    }
    last_remb_time_ = now;
    last_send_remb_bitrate_ = receive_bitrate;
    receive_bitrate = std::min(last_send_remb_bitrate_, max_remb_bitrate_);
  }
  remb_sender_(receive_bitrate.bps(), ssrcs);
}

void RembThrottler::SetMaxDesiredReceiveBitrate(DataRate bitrate) {
  Timestamp now = clock_->CurrentTime();
  {
    MutexLock lock(&mutex_);
    max_remb_bitrate_ = bitrate;
    if (now - last_remb_time_ < kRembSendInterval &&
        !last_send_remb_bitrate_.IsZero() &&
        last_send_remb_bitrate_ <= max_remb_bitrate_) {
      return;
    }
  }
  remb_sender_(bitrate.bps(), /*ssrcs=*/{});
}

}  // namespace webrtc
