/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#include "modules/rtp_rtcp/source/rtcp_transceiver_config.h"

#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "rtc_base/logging.h"

namespace webrtc {

RtcpTransceiverConfig::RtcpTransceiverConfig() = default;
RtcpTransceiverConfig::RtcpTransceiverConfig(const RtcpTransceiverConfig&) =
    default;
RtcpTransceiverConfig& RtcpTransceiverConfig::operator=(
    const RtcpTransceiverConfig&) = default;
RtcpTransceiverConfig::~RtcpTransceiverConfig() = default;

bool RtcpTransceiverConfig::Validate() const {
  if (feedback_ssrc == 0) {
    RTC_LOG(LS_WARNING)
        << debug_id
        << "Ssrc 0 may be treated by some implementation as invalid.";
  }
  if (cname.size() > 255) {
    RTC_LOG(LS_ERROR) << debug_id << "cname can be maximum 255 characters.";
    return false;
  }
  if (max_packet_size < 100) {
    RTC_LOG(LS_ERROR) << debug_id << "max packet size " << max_packet_size
                      << " is too small.";
    return false;
  }
  if (max_packet_size > IP_PACKET_SIZE) {
    RTC_LOG(LS_ERROR) << debug_id << "max packet size " << max_packet_size
                      << " more than " << IP_PACKET_SIZE << " is unsupported.";
    return false;
  }
  if (clock == nullptr) {
    RTC_LOG(LS_ERROR) << debug_id << "clock must be set";
    return false;
  }
  if (initial_report_delay < TimeDelta::Zero()) {
    RTC_LOG(LS_ERROR) << debug_id << "delay " << initial_report_delay.ms()
                      << "ms before first report shouldn't be negative.";
    return false;
  }
  if (report_period <= TimeDelta::Zero()) {
    RTC_LOG(LS_ERROR) << debug_id << "period " << report_period.ms()
                      << "ms between reports should be positive.";
    return false;
  }
  if (schedule_periodic_compound_packets && task_queue == nullptr) {
    RTC_LOG(LS_ERROR) << debug_id
                      << "missing task queue for periodic compound packets";
    return false;
  }
  if (rtcp_mode != RtcpMode::kCompound && rtcp_mode != RtcpMode::kReducedSize) {
    RTC_LOG(LS_ERROR) << debug_id << "unsupported rtcp mode";
    return false;
  }
  if (non_sender_rtt_measurement && !network_link_observer) {
    RTC_LOG(LS_WARNING) << debug_id
                        << "Enabled special feature to calculate rtt, but no "
                           "rtt observer is provided.";
  }
  return true;
}

}  // namespace webrtc
