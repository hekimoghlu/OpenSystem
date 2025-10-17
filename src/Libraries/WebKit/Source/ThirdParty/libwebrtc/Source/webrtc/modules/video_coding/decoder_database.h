/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#ifndef MODULES_VIDEO_CODING_DECODER_DATABASE_H_
#define MODULES_VIDEO_CODING_DECODER_DATABASE_H_

#include <stdint.h>

#include <map>
#include <memory>
#include <optional>

#include "api/sequence_checker.h"
#include "api/video/encoded_frame.h"
#include "api/video_codecs/video_decoder.h"
#include "modules/video_coding/generic_decoder.h"

namespace webrtc {

class VCMDecoderDatabase {
 public:
  VCMDecoderDatabase();
  VCMDecoderDatabase(const VCMDecoderDatabase&) = delete;
  VCMDecoderDatabase& operator=(const VCMDecoderDatabase&) = delete;
  ~VCMDecoderDatabase() = default;

  // Returns a pointer to the previously registered decoder or nullptr if none
  // was registered for the `payload_type`.
  void DeregisterExternalDecoder(uint8_t payload_type);
  void RegisterExternalDecoder(uint8_t payload_type,
                               std::unique_ptr<VideoDecoder> external_decoder);
  bool IsExternalDecoderRegistered(uint8_t payload_type) const;

  void RegisterReceiveCodec(uint8_t payload_type,
                            const VideoDecoder::Settings& settings);
  bool DeregisterReceiveCodec(uint8_t payload_type);
  void DeregisterReceiveCodecs();

  // Returns a decoder specified by frame.PayloadType. The decoded frame
  // callback of the decoder is set to `decoded_frame_callback`. If no such
  // decoder already exists an instance will be created and initialized.
  // nullptr is returned if no decoder with the specified payload type was found
  // and the function failed to create one.
  VCMGenericDecoder* GetDecoder(
      const EncodedFrame& frame,
      VCMDecodedFrameCallback* decoded_frame_callback);

 private:
  void CreateAndInitDecoder(const EncodedFrame& frame)
      RTC_RUN_ON(decoder_sequence_checker_);

  SequenceChecker decoder_sequence_checker_;

  std::optional<uint8_t> current_payload_type_;
  std::optional<VCMGenericDecoder> current_decoder_
      RTC_GUARDED_BY(decoder_sequence_checker_);
  // Initialization paramaters for decoders keyed by payload type.
  std::map<uint8_t, VideoDecoder::Settings> decoder_settings_;
  // Decoders keyed by payload type.
  std::map<uint8_t, std::unique_ptr<VideoDecoder>> decoders_
      RTC_GUARDED_BY(decoder_sequence_checker_);
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_DECODER_DATABASE_H_
