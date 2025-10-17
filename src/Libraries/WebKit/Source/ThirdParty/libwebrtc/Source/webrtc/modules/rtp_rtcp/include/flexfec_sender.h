/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
#ifndef MODULES_RTP_RTCP_INCLUDE_FLEXFEC_SENDER_H_
#define MODULES_RTP_RTCP_INCLUDE_FLEXFEC_SENDER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/environment/environment.h"
#include "api/rtp_parameters.h"
#include "api/units/timestamp.h"
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/rtp_header_extension_size.h"
#include "modules/rtp_rtcp/source/ulpfec_generator.h"
#include "modules/rtp_rtcp/source/video_fec_generator.h"
#include "rtc_base/bitrate_tracker.h"
#include "rtc_base/random.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

class RtpPacketToSend;

// Note that this class is not thread safe, and thus requires external
// synchronization. Currently, this is done using the lock in PayloadRouter.

class FlexfecSender : public VideoFecGenerator {
 public:
  FlexfecSender(const Environment& env,
                int payload_type,
                uint32_t ssrc,
                uint32_t protected_media_ssrc,
                absl::string_view mid,
                const std::vector<RtpExtension>& rtp_header_extensions,
                rtc::ArrayView<const RtpExtensionSize> extension_sizes,
                const RtpState* rtp_state);
  ~FlexfecSender();

  FecType GetFecType() const override {
    return VideoFecGenerator::FecType::kFlexFec;
  }
  std::optional<uint32_t> FecSsrc() override { return ssrc_; }

  // Sets the FEC rate, max frames sent before FEC packets are sent,
  // and what type of generator matrices are used.
  void SetProtectionParameters(const FecProtectionParams& delta_params,
                               const FecProtectionParams& key_params) override;

  // Adds a media packet to the internal buffer. When enough media packets
  // have been added, the FEC packets are generated and stored internally.
  // These FEC packets are then obtained by calling GetFecPackets().
  void AddPacketAndGenerateFec(const RtpPacketToSend& packet) override;

  // Returns generated FlexFEC packets.
  std::vector<std::unique_ptr<RtpPacketToSend>> GetFecPackets() override;

  // Returns the overhead, per packet, for FlexFEC.
  size_t MaxPacketOverhead() const override;

  DataRate CurrentFecRate() const override;

  // Only called on the VideoSendStream queue, after operation has shut down.
  std::optional<RtpState> GetRtpState() override;

 private:
  // Utility.
  const Environment env_;
  Random random_;
  Timestamp last_generated_packet_ = Timestamp::MinusInfinity();

  // Config.
  const int payload_type_;
  const uint32_t timestamp_offset_;
  const uint32_t ssrc_;
  const uint32_t protected_media_ssrc_;
  // MID value to send in the MID header extension.
  const std::string mid_;
  // Sequence number of next packet to generate.
  uint16_t seq_num_;

  // Implementation.
  UlpfecGenerator ulpfec_generator_;
  const RtpHeaderExtensionMap rtp_header_extension_map_;
  const size_t header_extensions_size_;

  mutable Mutex mutex_;
  BitrateTracker fec_bitrate_ RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_INCLUDE_FLEXFEC_SENDER_H_
