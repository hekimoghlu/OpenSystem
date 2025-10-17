/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
#include "api/audio_codecs/opus/audio_decoder_multi_channel_opus.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "api/audio_codecs/audio_codec_pair_id.h"
#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/audio_codecs/opus/audio_decoder_multi_channel_opus_config.h"
#include "api/field_trials_view.h"
#include "modules/audio_coding/codecs/opus/audio_decoder_multi_channel_opus_impl.h"

namespace webrtc {

std::optional<AudioDecoderMultiChannelOpusConfig>
AudioDecoderMultiChannelOpus::SdpToConfig(const SdpAudioFormat& format) {
  return AudioDecoderMultiChannelOpusImpl::SdpToConfig(format);
}

void AudioDecoderMultiChannelOpus::AppendSupportedDecoders(
    std::vector<AudioCodecSpec>* specs) {
  // To get full utilization of the surround support of the Opus lib, we can
  // mark which channel is the low frequency effects (LFE). But that is not done
  // ATM.
  {
    AudioCodecInfo surround_5_1_opus_info{48000, 6,
                                          /* default_bitrate_bps= */ 128000};
    surround_5_1_opus_info.allow_comfort_noise = false;
    surround_5_1_opus_info.supports_network_adaption = false;
    SdpAudioFormat opus_format({"multiopus",
                                48000,
                                6,
                                {{"minptime", "10"},
                                 {"useinbandfec", "1"},
                                 {"channel_mapping", "0,4,1,2,3,5"},
                                 {"num_streams", "4"},
                                 {"coupled_streams", "2"}}});
    specs->push_back({std::move(opus_format), surround_5_1_opus_info});
  }
  {
    AudioCodecInfo surround_7_1_opus_info{48000, 8,
                                          /* default_bitrate_bps= */ 200000};
    surround_7_1_opus_info.allow_comfort_noise = false;
    surround_7_1_opus_info.supports_network_adaption = false;
    SdpAudioFormat opus_format({"multiopus",
                                48000,
                                8,
                                {{"minptime", "10"},
                                 {"useinbandfec", "1"},
                                 {"channel_mapping", "0,6,1,2,3,4,5,7"},
                                 {"num_streams", "5"},
                                 {"coupled_streams", "3"}}});
    specs->push_back({std::move(opus_format), surround_7_1_opus_info});
  }
}

std::unique_ptr<AudioDecoder> AudioDecoderMultiChannelOpus::MakeAudioDecoder(
    AudioDecoderMultiChannelOpusConfig config,
    std::optional<AudioCodecPairId> /*codec_pair_id*/,
    const FieldTrialsView* /* field_trials */) {
  return AudioDecoderMultiChannelOpusImpl::MakeAudioDecoder(config);
}
}  // namespace webrtc
