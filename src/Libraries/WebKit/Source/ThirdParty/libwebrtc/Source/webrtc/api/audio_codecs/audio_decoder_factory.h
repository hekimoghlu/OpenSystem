/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
#ifndef API_AUDIO_CODECS_AUDIO_DECODER_FACTORY_H_
#define API_AUDIO_CODECS_AUDIO_DECODER_FACTORY_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"
#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/environment/environment.h"
#include "api/ref_count.h"

namespace webrtc {

// A factory that creates AudioDecoders.
class AudioDecoderFactory : public RefCountInterface {
 public:
  virtual std::vector<AudioCodecSpec> GetSupportedDecoders() = 0;

  virtual bool IsSupportedDecoder(const SdpAudioFormat& format) = 0;

  // Create a new decoder instance. The `codec_pair_id` argument is used to link
  // encoders and decoders that talk to the same remote entity: if a
  // AudioEncoderFactory::Create() and a AudioDecoderFactory::Create() call
  // receive non-null IDs that compare equal, the factory implementations may
  // assume that the encoder and decoder form a pair. (The intended use case for
  // this is to set up communication between the AudioEncoder and AudioDecoder
  // instances, which is needed for some codecs with built-in bandwidth
  // adaptation.)
  //
  // Returns null if the format isn't supported.
  //
  // Note: Implementations need to be robust against combinations other than
  // one encoder, one decoder getting the same ID; such decoders must still
  // work.
  virtual absl::Nullable<std::unique_ptr<AudioDecoder>> Create(
      const Environment& env,
      const SdpAudioFormat& format,
      std::optional<AudioCodecPairId> codec_pair_id) = 0;
};

}  // namespace webrtc

#endif  // API_AUDIO_CODECS_AUDIO_DECODER_FACTORY_H_
