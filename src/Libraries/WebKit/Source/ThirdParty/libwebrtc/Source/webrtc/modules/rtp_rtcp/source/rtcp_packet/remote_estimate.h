/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_REMOTE_ESTIMATE_H_
#define MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_REMOTE_ESTIMATE_H_

#include <memory>
#include <vector>

#include "api/transport/network_types.h"
#include "modules/rtp_rtcp/source/rtcp_packet/app.h"

namespace webrtc {
namespace rtcp {

class CommonHeader;
class RemoteEstimateSerializer {
 public:
  virtual bool Parse(rtc::ArrayView<const uint8_t> src,
                     NetworkStateEstimate* target) const = 0;
  virtual rtc::Buffer Serialize(const NetworkStateEstimate& src) const = 0;
  virtual ~RemoteEstimateSerializer() = default;
};

// Using a static global implementation to avoid incurring initialization
// overhead of the serializer every time RemoteEstimate is created.
const RemoteEstimateSerializer* GetRemoteEstimateSerializer();

// The RemoteEstimate packet provides network estimation results from the
// receive side. This functionality is experimental and subject to change
// without notice.
class RemoteEstimate : public App {
 public:
  RemoteEstimate();
  explicit RemoteEstimate(App&& app);
  // Note, sub type must be unique among all app messages with "goog" name.
  static constexpr uint8_t kSubType = 13;
  static constexpr uint32_t kName = NameToInt("goog");
  static TimeDelta GetTimestampPeriod();

  bool ParseData();
  void SetEstimate(NetworkStateEstimate estimate);
  NetworkStateEstimate estimate() const { return estimate_; }

 private:
  NetworkStateEstimate estimate_;
  const RemoteEstimateSerializer* const serializer_;
};

}  // namespace rtcp
}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTCP_PACKET_REMOTE_ESTIMATE_H_
