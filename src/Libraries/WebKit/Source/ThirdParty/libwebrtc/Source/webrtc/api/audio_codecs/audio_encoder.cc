/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#include "api/audio_codecs/audio_encoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "api/array_view.h"
#include "api/call/bitrate_allocation.h"
#include "rtc_base/buffer.h"
#include "rtc_base/checks.h"
#include "rtc_base/trace_event.h"

namespace webrtc {

ANAStats::ANAStats() = default;
ANAStats::~ANAStats() = default;
ANAStats::ANAStats(const ANAStats&) = default;

AudioEncoder::EncodedInfo::EncodedInfo() = default;
AudioEncoder::EncodedInfo::EncodedInfo(const EncodedInfo&) = default;
AudioEncoder::EncodedInfo::EncodedInfo(EncodedInfo&&) = default;
AudioEncoder::EncodedInfo::~EncodedInfo() = default;
AudioEncoder::EncodedInfo& AudioEncoder::EncodedInfo::operator=(
    const EncodedInfo&) = default;
AudioEncoder::EncodedInfo& AudioEncoder::EncodedInfo::operator=(EncodedInfo&&) =
    default;

int AudioEncoder::RtpTimestampRateHz() const {
  return SampleRateHz();
}

AudioEncoder::EncodedInfo AudioEncoder::Encode(
    uint32_t rtp_timestamp,
    rtc::ArrayView<const int16_t> audio,
    rtc::Buffer* encoded) {
  TRACE_EVENT0("webrtc", "AudioEncoder::Encode");
  RTC_CHECK_EQ(audio.size(),
               static_cast<size_t>(NumChannels() * SampleRateHz() / 100));

  const size_t old_size = encoded->size();
  EncodedInfo info = EncodeImpl(rtp_timestamp, audio, encoded);
  RTC_CHECK_EQ(encoded->size() - old_size, info.encoded_bytes);
  return info;
}

bool AudioEncoder::SetFec(bool enable) {
  return !enable;
}

bool AudioEncoder::SetDtx(bool enable) {
  return !enable;
}

bool AudioEncoder::GetDtx() const {
  return false;
}

bool AudioEncoder::SetApplication(Application /* application */) {
  return false;
}

void AudioEncoder::SetMaxPlaybackRate(int /* frequency_hz */) {}

void AudioEncoder::SetTargetBitrate(int /* target_bps */) {}

rtc::ArrayView<std::unique_ptr<AudioEncoder>>
AudioEncoder::ReclaimContainedEncoders() {
  return nullptr;
}

bool AudioEncoder::EnableAudioNetworkAdaptor(
    const std::string& /* config_string */,
    RtcEventLog* /* event_log */) {
  return false;
}

void AudioEncoder::DisableAudioNetworkAdaptor() {}

void AudioEncoder::OnReceivedUplinkPacketLossFraction(
    float /* uplink_packet_loss_fraction */) {}

void AudioEncoder::OnReceivedUplinkRecoverablePacketLossFraction(
    float /* uplink_recoverable_packet_loss_fraction */) {
  RTC_DCHECK_NOTREACHED();
}

void AudioEncoder::OnReceivedTargetAudioBitrate(int target_audio_bitrate_bps) {
  OnReceivedUplinkBandwidth(target_audio_bitrate_bps, std::nullopt);
}

void AudioEncoder::OnReceivedUplinkBandwidth(
    int /* target_audio_bitrate_bps */,
    std::optional<int64_t> /* bwe_period_ms */) {}

void AudioEncoder::OnReceivedUplinkAllocation(BitrateAllocationUpdate update) {
  OnReceivedUplinkBandwidth(update.target_bitrate.bps(),
                            update.bwe_period.ms());
}

void AudioEncoder::OnReceivedRtt(int /* rtt_ms */) {}

void AudioEncoder::OnReceivedOverhead(size_t /* overhead_bytes_per_packet */) {}

void AudioEncoder::SetReceiverFrameLengthRange(int /* min_frame_length_ms */,
                                               int /* max_frame_length_ms */) {}

ANAStats AudioEncoder::GetANAStats() const {
  return ANAStats();
}

constexpr int AudioEncoder::kMaxNumberOfChannels;
}  // namespace webrtc
