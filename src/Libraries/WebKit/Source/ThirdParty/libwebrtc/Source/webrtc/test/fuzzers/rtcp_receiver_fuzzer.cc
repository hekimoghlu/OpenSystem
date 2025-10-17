/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include <cstddef>
#include <cstdint>
#include <vector>

#include "api/array_view.h"
#include "api/environment/environment_factory.h"
#include "modules/rtp_rtcp/include/report_block_data.h"
#include "modules/rtp_rtcp/source/rtcp_packet/tmmb_item.h"
#include "modules/rtp_rtcp/source/rtcp_receiver.h"
#include "modules/rtp_rtcp/source/rtp_rtcp_interface.h"
#include "system_wrappers/include/clock.h"
#include "test/explicit_key_value_config.h"

namespace webrtc {
namespace {

constexpr int kRtcpIntervalMs = 1000;

// RTCP is typically sent over UDP, which has a maximum payload length
// of 65535 bytes. We err on the side of caution and check a bit above that.
constexpr size_t kMaxInputLenBytes = 66000;

class NullModuleRtpRtcp : public RTCPReceiver::ModuleRtpRtcp {
 public:
  void SetTmmbn(std::vector<rtcp::TmmbItem>) override {}
  void OnRequestSendReport() override {}
  void OnReceivedNack(const std::vector<uint16_t>&) override {}
  void OnReceivedRtcpReportBlocks(
      rtc::ArrayView<const ReportBlockData> report_blocks) override {}
};

}  // namespace

void FuzzOneInput(const uint8_t* data, size_t size) {
  if (size > kMaxInputLenBytes) {
    return;
  }
  test::ExplicitKeyValueConfig field_trials(
      "WebRTC-RFC8888CongestionControlFeedback/Enabled/");
  NullModuleRtpRtcp rtp_rtcp_module;
  SimulatedClock clock(1234);

  RtpRtcpInterface::Configuration config;
  config.rtcp_report_interval_ms = kRtcpIntervalMs;
  config.local_media_ssrc = 1;

  RTCPReceiver receiver(CreateEnvironment(&clock, &field_trials), config,
                        &rtp_rtcp_module);

  receiver.IncomingPacket(rtc::MakeArrayView(data, size));
}
}  // namespace webrtc
