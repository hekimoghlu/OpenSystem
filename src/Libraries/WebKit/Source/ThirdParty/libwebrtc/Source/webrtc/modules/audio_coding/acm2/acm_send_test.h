/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#ifndef MODULES_AUDIO_CODING_ACM2_ACM_SEND_TEST_H_
#define MODULES_AUDIO_CODING_ACM2_ACM_SEND_TEST_H_

#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/audio/audio_frame.h"
#include "api/environment/environment.h"
#include "modules/audio_coding/include/audio_coding_module.h"
#include "modules/audio_coding/neteq/tools/packet_source.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {
class AudioEncoder;

namespace test {
class InputAudioFile;
class Packet;

class AcmSendTestOldApi : public AudioPacketizationCallback,
                          public PacketSource {
 public:
  AcmSendTestOldApi(InputAudioFile* audio_source,
                    int source_rate_hz,
                    int test_duration_ms);
  ~AcmSendTestOldApi() override;

  AcmSendTestOldApi(const AcmSendTestOldApi&) = delete;
  AcmSendTestOldApi& operator=(const AcmSendTestOldApi&) = delete;

  // Registers the send codec. Returns true on success, false otherwise.
  bool RegisterCodec(absl::string_view payload_name,
                     int sampling_freq_hz,
                     int channels,
                     int payload_type,
                     int frame_size_samples);

  // Registers an external send codec.
  void RegisterExternalCodec(
      std::unique_ptr<AudioEncoder> external_speech_encoder);

  // Inherited from PacketSource.
  std::unique_ptr<Packet> NextPacket() override;

  // Inherited from AudioPacketizationCallback.
  int32_t SendData(AudioFrameType frame_type,
                   uint8_t payload_type,
                   uint32_t timestamp,
                   const uint8_t* payload_data,
                   size_t payload_len_bytes,
                   int64_t absolute_capture_timestamp_ms) override;

  AudioCodingModule* acm() { return acm_.get(); }

 private:
  static const int kBlockSizeMs = 10;

  // Creates a Packet object from the last packet produced by ACM (and received
  // through the SendData method as a callback).
  std::unique_ptr<Packet> CreatePacket();

  SimulatedClock clock_;
  const Environment env_;
  std::unique_ptr<AudioCodingModule> acm_;
  InputAudioFile* audio_source_;
  int source_rate_hz_;
  const size_t input_block_size_samples_;
  AudioFrame input_frame_;
  bool codec_registered_;
  int test_duration_ms_;
  // The following member variables are set whenever SendData() is called.
  AudioFrameType frame_type_;
  int payload_type_;
  uint32_t timestamp_;
  uint16_t sequence_number_;
  std::vector<uint8_t> last_payload_vec_;
  bool data_to_send_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_ACM2_ACM_SEND_TEST_H_
