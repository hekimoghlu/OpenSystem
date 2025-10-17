/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#ifndef API_VOIP_VOIP_CODEC_H_
#define API_VOIP_VOIP_CODEC_H_

#include <map>

#include "api/audio_codecs/audio_format.h"
#include "api/voip/voip_base.h"

namespace webrtc {

// VoipCodec interface currently provides any codec related interface
// such as setting encoder and decoder types that are negotiated with
// remote endpoint.  Typically after SDP offer and answer exchange,
// the local endpoint understands what are the codec payload types that
// are used with negotiated codecs.  This interface is subject to expand
// as needed in future.
//
// This interface requires a channel id created via VoipBase interface.
class VoipCodec {
 public:
  // Set encoder type here along with its payload type to use.
  // Returns following VoipResult;
  //  kOk - sending codec is set as provided.
  //  kInvalidArgument - `channel_id` is invalid.
  virtual VoipResult SetSendCodec(ChannelId channel_id,
                                  int payload_type,
                                  const SdpAudioFormat& encoder_spec) = 0;

  // Set decoder payload type here. In typical offer and answer model,
  // this should be called after payload type has been agreed in media
  // session.  Note that payload type can differ with same codec in each
  // direction.
  // Returns following VoipResult;
  //  kOk - receiving codecs are set as provided.
  //  kInvalidArgument - `channel_id` is invalid.
  virtual VoipResult SetReceiveCodecs(
      ChannelId channel_id,
      const std::map<int, SdpAudioFormat>& decoder_specs) = 0;

 protected:
  virtual ~VoipCodec() = default;
};

}  // namespace webrtc

#endif  // API_VOIP_VOIP_CODEC_H_
