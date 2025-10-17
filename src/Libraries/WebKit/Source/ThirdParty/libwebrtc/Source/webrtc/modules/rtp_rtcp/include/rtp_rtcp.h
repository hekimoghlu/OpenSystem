/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#ifndef MODULES_RTP_RTCP_INCLUDE_RTP_RTCP_H_
#define MODULES_RTP_RTCP_INCLUDE_RTP_RTCP_H_

#include <memory>

#include "api/environment/environment.h"
#include "modules/rtp_rtcp/source/rtp_rtcp_interface.h"

namespace webrtc {

// A deprecated version of the RtpRtcp module.
class [[deprecated("bugs.webrtc.org/42224904")]] RtpRtcp
    : public RtpRtcpInterface {
 public:
  [[deprecated("bugs.webrtc.org/42224904")]]  //
  static std::unique_ptr<RtpRtcp>
  Create(const Environment& env, const Configuration& configuration);

  // Process any pending tasks such as timeouts.
  virtual void Process() = 0;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_INCLUDE_RTP_RTCP_H_
