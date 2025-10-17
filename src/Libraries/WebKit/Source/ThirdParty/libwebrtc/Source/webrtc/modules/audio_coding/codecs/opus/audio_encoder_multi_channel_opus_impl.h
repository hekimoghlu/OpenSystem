/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#ifndef MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_ENCODER_MULTI_CHANNEL_OPUS_IMPL_H_
#define MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_ENCODER_MULTI_CHANNEL_OPUS_IMPL_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "api/array_view.h"
#include "api/audio_codecs/audio_encoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/audio_codecs/opus/audio_encoder_multi_channel_opus_config.h"
#include "api/units/time_delta.h"
#include "modules/audio_coding/codecs/opus/opus_interface.h"
#include "rtc_base/buffer.h"

namespace webrtc {

class RtcEventLog;

class AudioEncoderMultiChannelOpusImpl final : public AudioEncoder {
 public:
  AudioEncoderMultiChannelOpusImpl(
      const AudioEncoderMultiChannelOpusConfig& config,
      int payload_type);
  ~AudioEncoderMultiChannelOpusImpl() override;

  AudioEncoderMultiChannelOpusImpl(const AudioEncoderMultiChannelOpusImpl&) =
      delete;
  AudioEncoderMultiChannelOpusImpl& operator=(
      const AudioEncoderMultiChannelOpusImpl&) = delete;

  // Static interface for use by BuiltinAudioEncoderFactory.
  static constexpr const char* GetPayloadName() { return "multiopus"; }
  static std::optional<AudioCodecInfo> QueryAudioEncoder(
      const SdpAudioFormat& format);

  int SampleRateHz() const override;
  size_t NumChannels() const override;
  size_t Num10MsFramesInNextPacket() const override;
  size_t Max10MsFramesInAPacket() const override;
  int GetTargetBitrate() const override;

  void Reset() override;
  std::optional<std::pair<TimeDelta, TimeDelta>> GetFrameLengthRange()
      const override;

 protected:
  EncodedInfo EncodeImpl(uint32_t rtp_timestamp,
                         rtc::ArrayView<const int16_t> audio,
                         rtc::Buffer* encoded) override;

 private:
  static std::optional<AudioEncoderMultiChannelOpusConfig> SdpToConfig(
      const SdpAudioFormat& format);
  static AudioCodecInfo QueryAudioEncoder(
      const AudioEncoderMultiChannelOpusConfig& config);
  static std::unique_ptr<AudioEncoder> MakeAudioEncoder(
      const AudioEncoderMultiChannelOpusConfig&,
      int payload_type);

  size_t Num10msFramesPerPacket() const;
  size_t SamplesPer10msFrame() const;
  size_t SufficientOutputBufferSize() const;
  bool RecreateEncoderInstance(
      const AudioEncoderMultiChannelOpusConfig& config);
  void SetFrameLength(int frame_length_ms);
  void SetNumChannelsToEncode(size_t num_channels_to_encode);
  void SetProjectedPacketLossRate(float fraction);

  AudioEncoderMultiChannelOpusConfig config_;
  const int payload_type_;
  std::vector<int16_t> input_buffer_;
  OpusEncInst* inst_;
  uint32_t first_timestamp_in_buffer_;
  size_t num_channels_to_encode_;
  int next_frame_length_ms_;

  friend struct AudioEncoderMultiChannelOpus;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_ENCODER_MULTI_CHANNEL_OPUS_IMPL_H_
