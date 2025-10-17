/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#ifndef MODULES_AUDIO_CODING_CODECS_G722_AUDIO_ENCODER_G722_H_
#define MODULES_AUDIO_CODING_CODECS_G722_AUDIO_ENCODER_G722_H_

#include <memory>
#include <optional>
#include <utility>

#include "api/audio_codecs/audio_encoder.h"
#include "api/audio_codecs/g722/audio_encoder_g722_config.h"
#include "api/units/time_delta.h"
#include "modules/audio_coding/codecs/g722/g722_interface.h"
#include "rtc_base/buffer.h"

namespace webrtc {

class AudioEncoderG722Impl final : public AudioEncoder {
 public:
  AudioEncoderG722Impl(const AudioEncoderG722Config& config, int payload_type);
  ~AudioEncoderG722Impl() override;

  AudioEncoderG722Impl(const AudioEncoderG722Impl&) = delete;
  AudioEncoderG722Impl& operator=(const AudioEncoderG722Impl&) = delete;

  int SampleRateHz() const override;
  size_t NumChannels() const override;
  int RtpTimestampRateHz() const override;
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
  // The encoder state for one channel.
  struct EncoderState {
    G722EncInst* encoder;
    std::unique_ptr<int16_t[]> speech_buffer;  // Queued up for encoding.
    rtc::Buffer encoded_buffer;                // Already encoded.
    EncoderState();
    ~EncoderState();
  };

  size_t SamplesPerChannel() const;

  const size_t num_channels_;
  const int payload_type_;
  const size_t num_10ms_frames_per_packet_;
  size_t num_10ms_frames_buffered_;
  uint32_t first_timestamp_in_buffer_;
  const std::unique_ptr<EncoderState[]> encoders_;
  rtc::Buffer interleave_buffer_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_CODECS_G722_AUDIO_ENCODER_G722_H_
