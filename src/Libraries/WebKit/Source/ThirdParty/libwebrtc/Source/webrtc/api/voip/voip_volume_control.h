/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#ifndef API_VOIP_VOIP_VOLUME_CONTROL_H_
#define API_VOIP_VOIP_VOLUME_CONTROL_H_

#include "api/voip/voip_base.h"

namespace webrtc {

struct VolumeInfo {
  // https://w3c.github.io/webrtc-stats/#dom-rtcaudiosourcestats-audiolevel
  double audio_level = 0;
  // https://w3c.github.io/webrtc-stats/#dom-rtcaudiosourcestats-totalaudioenergy
  double total_energy = 0.0;
  // https://w3c.github.io/webrtc-stats/#dom-rtcaudiosourcestats-totalsamplesduration
  double total_duration = 0.0;
};

// VoipVolumeControl interface.
//
// This sub-API supports functions related to the input (microphone) and output
// (speaker) device.
//
// Caller must ensure that ChannelId is valid otherwise it will result in no-op
// with error logging.
class VoipVolumeControl {
 public:
  // Mute/unmutes the microphone input sample before encoding process. Note that
  // mute doesn't affect audio input level and energy values as input sample is
  // silenced after the measurement.
  // Returns following VoipResult;
  //  kOk - input source muted or unmuted as provided by `enable`.
  //  kInvalidArgument - `channel_id` is invalid.
  virtual VoipResult SetInputMuted(ChannelId channel_id, bool enable) = 0;

  // Gets the microphone volume info via `volume_info` reference.
  // Returns following VoipResult;
  //  kOk - successfully set provided input volume info.
  //  kInvalidArgument - `channel_id` is invalid.
  virtual VoipResult GetInputVolumeInfo(ChannelId channel_id,
                                        VolumeInfo& volume_info) = 0;

  // Gets the speaker volume info via `volume_info` reference.
  // Returns following VoipResult;
  //  kOk - successfully set provided output volume info.
  //  kInvalidArgument - `channel_id` is invalid.
  virtual VoipResult GetOutputVolumeInfo(ChannelId channel_id,
                                         VolumeInfo& volume_info) = 0;

 protected:
  virtual ~VoipVolumeControl() = default;
};

}  // namespace webrtc

#endif  // API_VOIP_VOIP_VOLUME_CONTROL_H_
