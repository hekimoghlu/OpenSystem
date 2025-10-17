/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#include "net/dcsctp/tx/retransmission_timeout.h"

#include <algorithm>
#include <cstdint>

#include "api/units/time_delta.h"
#include "net/dcsctp/public/dcsctp_options.h"

namespace dcsctp {

// https://datatracker.ietf.org/doc/html/rfc4960#section-15.
constexpr double kRtoAlpha = 0.125;
constexpr double kRtoBeta = 0.25;

// A factor that the `min_rtt_variance` configuration option will be divided by
// (before later multiplied with K, which is 4 according to RFC6298). When this
// value was introduced, it was unintentionally divided by 8 since that code
// worked with scaled numbers (to avoid floating point math). That behavior is
// kept as downstream users have measured good values for their use-cases.
constexpr double kHeuristicVarianceAdjustment = 8.0;

RetransmissionTimeout::RetransmissionTimeout(const DcSctpOptions& options)
    : min_rto_(options.rto_min.ToTimeDelta()),
      max_rto_(options.rto_max.ToTimeDelta()),
      max_rtt_(options.rtt_max.ToTimeDelta()),
      min_rtt_variance_(options.min_rtt_variance.ToTimeDelta() /
                        kHeuristicVarianceAdjustment),
      srtt_(options.rto_initial.ToTimeDelta()),
      rto_(options.rto_initial.ToTimeDelta()) {}

void RetransmissionTimeout::ObserveRTT(webrtc::TimeDelta rtt) {
  // Unrealistic values will be skipped. If a wrongly measured (or otherwise
  // corrupt) value was processed, it could change the state in a way that would
  // take a very long time to recover.
  if (rtt < webrtc::TimeDelta::Zero() || rtt > max_rtt_) {
    return;
  }

  // https://tools.ietf.org/html/rfc4960#section-6.3.1.
  if (first_measurement_) {
    srtt_ = rtt;
    rtt_var_ = rtt / 2;
    first_measurement_ = false;
  } else {
    webrtc::TimeDelta rtt_diff = (srtt_ - rtt).Abs();
    rtt_var_ = (1 - kRtoBeta) * rtt_var_ + kRtoBeta * rtt_diff;
    srtt_ = (1 - kRtoAlpha) * srtt_ + kRtoAlpha * rtt;
  }

  if (rtt_var_ < min_rtt_variance_) {
    rtt_var_ = min_rtt_variance_;
  }

  rto_ = srtt_ + 4 * rtt_var_;
  rto_ = std::clamp(rto_, min_rto_, max_rto_);
}
}  // namespace dcsctp
