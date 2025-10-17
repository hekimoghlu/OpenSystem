/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
#ifndef API_AUDIO_CODECS_OPUS_AUDIO_DECODER_MULTI_CHANNEL_OPUS_CONFIG_H_
#define API_AUDIO_CODECS_OPUS_AUDIO_DECODER_MULTI_CHANNEL_OPUS_CONFIG_H_

#include <vector>

#include "api/audio_codecs/audio_decoder.h"

namespace webrtc {
struct AudioDecoderMultiChannelOpusConfig {
  // The number of channels that the decoder will output.
  int num_channels;

  // Number of mono or stereo encoded Opus streams.
  int num_streams;

  // Number of channel pairs coupled together, see RFC 7845 section
  // 5.1.1. Has to be less than the number of streams.
  int coupled_streams;

  // Channel mapping table, defines the mapping from encoded streams to output
  // channels. See RFC 7845 section 5.1.1.
  std::vector<unsigned char> channel_mapping;

  bool IsOk() const {
    if (num_channels < 1 || num_channels > AudioDecoder::kMaxNumberOfChannels ||
        num_streams < 0 || coupled_streams < 0) {
      return false;
    }
    if (num_streams < coupled_streams) {
      return false;
    }
    if (channel_mapping.size() != static_cast<size_t>(num_channels)) {
      return false;
    }

    // Every mono stream codes one channel, every coupled stream codes two. This
    // is the total coded channel count:
    const int max_coded_channel = num_streams + coupled_streams;
    for (const auto& x : channel_mapping) {
      // Coded channels >= max_coded_channel don't exist. Except for 255, which
      // tells Opus to put silence in output channel x.
      if (x >= max_coded_channel && x != 255) {
        return false;
      }
    }

    if (num_channels > 255 || max_coded_channel >= 255) {
      return false;
    }
    return true;
  }
};

}  // namespace webrtc

#endif  //  API_AUDIO_CODECS_OPUS_AUDIO_DECODER_MULTI_CHANNEL_OPUS_CONFIG_H_
