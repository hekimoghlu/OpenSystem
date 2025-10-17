/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#include "modules/audio_coding/neteq/tools/neteq_rtp_dump_input.h"

#include "absl/strings/string_view.h"
#include "modules/audio_coding/neteq/tools/rtp_file_source.h"

namespace webrtc {
namespace test {
namespace {

// An adapter class to dress up a PacketSource object as a NetEqInput.
class NetEqRtpDumpInput : public NetEqInput {
 public:
  NetEqRtpDumpInput(absl::string_view file_name,
                    const std::map<int, RTPExtensionType>& hdr_ext_map,
                    std::optional<uint32_t> ssrc_filter)
      : source_(RtpFileSource::Create(file_name, ssrc_filter)) {
    for (const auto& ext_pair : hdr_ext_map) {
      source_->RegisterRtpHeaderExtension(ext_pair.second, ext_pair.first);
    }
    LoadNextPacket();
  }

  std::optional<int64_t> NextOutputEventTime() const override {
    return next_output_event_ms_;
  }

  std::optional<SetMinimumDelayInfo> NextSetMinimumDelayInfo() const override {
    return std::nullopt;
  }

  void AdvanceOutputEvent() override {
    if (next_output_event_ms_) {
      *next_output_event_ms_ += kOutputPeriodMs;
    }
    if (!NextPacketTime()) {
      next_output_event_ms_ = std::nullopt;
    }
  }

  void AdvanceSetMinimumDelay() override {}

  std::optional<int64_t> NextPacketTime() const override {
    return packet_ ? std::optional<int64_t>(
                         static_cast<int64_t>(packet_->time_ms()))
                   : std::nullopt;
  }

  std::unique_ptr<PacketData> PopPacket() override {
    if (!packet_) {
      return std::unique_ptr<PacketData>();
    }
    std::unique_ptr<PacketData> packet_data(new PacketData);
    packet_data->header = packet_->header();
    if (packet_->payload_length_bytes() == 0 &&
        packet_->virtual_payload_length_bytes() > 0) {
      // This is a header-only "dummy" packet. Set the payload to all zeros,
      // with length according to the virtual length.
      packet_data->payload.SetSize(packet_->virtual_payload_length_bytes());
      std::fill_n(packet_data->payload.data(), packet_data->payload.size(), 0);
    } else {
      packet_data->payload.SetData(packet_->payload(),
                                   packet_->payload_length_bytes());
    }
    packet_data->time_ms = packet_->time_ms();

    LoadNextPacket();

    return packet_data;
  }

  std::optional<RTPHeader> NextHeader() const override {
    return packet_ ? std::optional<RTPHeader>(packet_->header()) : std::nullopt;
  }

  bool ended() const override { return !next_output_event_ms_; }

 private:
  void LoadNextPacket() { packet_ = source_->NextPacket(); }

  std::optional<int64_t> next_output_event_ms_ = 0;
  static constexpr int64_t kOutputPeriodMs = 10;

  std::unique_ptr<RtpFileSource> source_;
  std::unique_ptr<Packet> packet_;
};

}  // namespace

std::unique_ptr<NetEqInput> CreateNetEqRtpDumpInput(
    absl::string_view file_name,
    const std::map<int, RTPExtensionType>& hdr_ext_map,
    std::optional<uint32_t> ssrc_filter) {
  return std::make_unique<NetEqRtpDumpInput>(file_name, hdr_ext_map,
                                             ssrc_filter);
}

}  // namespace test
}  // namespace webrtc
