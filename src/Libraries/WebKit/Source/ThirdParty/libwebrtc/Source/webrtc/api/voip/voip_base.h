/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
#ifndef API_VOIP_VOIP_BASE_H_
#define API_VOIP_VOIP_BASE_H_

#include <cstdint>
#include <optional>

#include "absl/base/attributes.h"

namespace webrtc {

class Transport;

// VoipBase interface
//
// VoipBase provides a management interface on a media session using a
// concept called 'channel'.  A channel represents an interface handle
// for application to request various media session operations.  This
// notion of channel is used throughout other interfaces as well.
//
// Underneath the interface, a channel id is mapped into an audio session
// object that is capable of sending and receiving a single RTP stream with
// another media endpoint.  It's possible to create and use multiple active
// channels simultaneously which would mean that particular application
// session has RTP streams with multiple remote endpoints.
//
// A typical example for the usage context is outlined in VoipEngine
// header file.

enum class ChannelId : int {};

enum class ABSL_MUST_USE_RESULT VoipResult {
  // kOk indicates the function was successfully invoked with no error.
  kOk,
  // kInvalidArgument indicates the caller specified an invalid argument, such
  // as an invalid ChannelId.
  kInvalidArgument,
  // kFailedPrecondition indicates that the operation was failed due to not
  // satisfying prerequisite such as not setting codec type before sending.
  kFailedPrecondition,
  // kInternal is used to indicate various internal failures that are not the
  // caller's fault. Further detail is commented on each function that uses this
  // return value.
  kInternal,
};

class VoipBase {
 public:
  // Creates a channel.
  // Each channel handle maps into one audio media session where each has
  // its own separate module for send/receive rtp packet with one peer.
  // Caller must set `transport`, webrtc::Transport callback pointer to
  // receive rtp/rtcp packets from corresponding media session in VoIP engine.
  // VoipEngine framework expects applications to handle network I/O directly
  // and injection for incoming RTP from remote endpoint is handled via
  // VoipNetwork interface. `local_ssrc` is optional and when local_ssrc is not
  // set, some random value will be used by voip engine.
  // Returns a ChannelId created for caller to handle subsequent Channel
  // operations.
  virtual ChannelId CreateChannel(Transport* transport,
                                  std::optional<uint32_t> local_ssrc) = 0;

  // Releases `channel_id` that no longer has any use.
  // Returns following VoipResult;
  //  kOk - `channel_id` is released.
  //  kInvalidArgument - `channel_id` is invalid.
  //  kInternal - Fails to stop audio output device.
  virtual VoipResult ReleaseChannel(ChannelId channel_id) = 0;

  // Starts sending on `channel_id`. This starts microphone if not started yet.
  // Returns following VoipResult;
  //  kOk - Channel successfully started to send.
  //  kInvalidArgument - `channel_id` is invalid.
  //  kFailedPrecondition - Missing prerequisite on VoipCodec::SetSendCodec.
  //  kInternal - initialization has failed on selected microphone.
  virtual VoipResult StartSend(ChannelId channel_id) = 0;

  // Stops sending on `channel_id`. If this is the last active channel, it will
  // stop microphone input from underlying audio platform layer.
  // Returns following VoipResult;
  //  kOk - Channel successfully stopped to send.
  //  kInvalidArgument - `channel_id` is invalid.
  //  kInternal - Failed to stop the active microphone device.
  virtual VoipResult StopSend(ChannelId channel_id) = 0;

  // Starts playing on speaker device for `channel_id`.
  // This will start underlying platform speaker device if not started.
  // Returns following VoipResult;
  //  kOk - Channel successfully started to play out.
  //  kInvalidArgument - `channel_id` is invalid.
  //  kFailedPrecondition - Missing prerequisite on VoipCodec::SetReceiveCodecs.
  //  kInternal - Failed to initializate the selected speaker device.
  virtual VoipResult StartPlayout(ChannelId channel_id) = 0;

  // Stops playing on speaker device for `channel_id`.
  // Returns following VoipResult;
  //  kOk - Channel successfully stopped t play out.
  //  kInvalidArgument - `channel_id` is invalid.
  virtual VoipResult StopPlayout(ChannelId channel_id) = 0;

 protected:
  virtual ~VoipBase() = default;
};

}  // namespace webrtc

#endif  // API_VOIP_VOIP_BASE_H_
