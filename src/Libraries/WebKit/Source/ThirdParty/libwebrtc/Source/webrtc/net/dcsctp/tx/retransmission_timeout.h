/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#ifndef NET_DCSCTP_TX_RETRANSMISSION_TIMEOUT_H_
#define NET_DCSCTP_TX_RETRANSMISSION_TIMEOUT_H_

#include <cstdint>
#include <functional>

#include "net/dcsctp/public/dcsctp_options.h"

namespace dcsctp {

// Manages updating of the Retransmission Timeout (RTO) SCTP variable, which is
// used directly as the base timeout for T3-RTX and for other timers, such as
// delayed ack.
//
// When a round-trip-time (RTT) is calculated (outside this class), `Observe`
// is called, which calculates the retransmission timeout (RTO) value. The RTO
// value will become larger if the RTT is high and/or the RTT values are varying
// a lot, which is an indicator of a bad connection.
class RetransmissionTimeout {
 public:
  explicit RetransmissionTimeout(const DcSctpOptions& options);

  // To be called when a RTT has been measured, to update the RTO value.
  void ObserveRTT(webrtc::TimeDelta rtt);

  // Returns the Retransmission Timeout (RTO) value.
  webrtc::TimeDelta rto() const { return rto_; }

  // Returns the smoothed RTT value.
  webrtc::TimeDelta srtt() const { return srtt_; }

 private:
  const webrtc::TimeDelta min_rto_;
  const webrtc::TimeDelta max_rto_;
  const webrtc::TimeDelta max_rtt_;
  const webrtc::TimeDelta min_rtt_variance_;
  // If this is the first measurement
  bool first_measurement_ = true;
  // Smoothed Round-Trip Time.
  webrtc::TimeDelta srtt_;
  // Round-Trip Time Variation.
  webrtc::TimeDelta rtt_var_ = webrtc::TimeDelta::Zero();
  // Retransmission Timeout
  webrtc::TimeDelta rto_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_TX_RETRANSMISSION_TIMEOUT_H_
