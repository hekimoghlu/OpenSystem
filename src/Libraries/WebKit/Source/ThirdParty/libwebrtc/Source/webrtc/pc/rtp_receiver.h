/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
// This file contains classes that implement RtpReceiverInterface.
// An RtpReceiver associates a MediaStreamTrackInterface with an underlying
// transport (provided by cricket::VoiceChannel/cricket::VideoChannel)

#ifndef PC_RTP_RECEIVER_H_
#define PC_RTP_RECEIVER_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "api/dtls_transport_interface.h"
#include "api/media_stream_interface.h"
#include "api/rtp_receiver_interface.h"
#include "api/scoped_refptr.h"
#include "media/base/media_channel.h"

namespace webrtc {

// Internal class used by PeerConnection.
class RtpReceiverInternal : public RtpReceiverInterface {
 public:
  // Call on the signaling thread, to let the receiver know that the the
  // embedded source object should enter a stopped/ended state and the track's
  // state set to `kEnded`, a final state that cannot be reversed.
  virtual void Stop() = 0;

  // Sets the underlying MediaEngine channel associated with this RtpSender.
  // A VoiceMediaChannel should be used for audio RtpSenders and
  // a VideoMediaChannel should be used for video RtpSenders.
  // NOTE:
  // * SetMediaChannel(nullptr) must be called before the media channel is
  //   destroyed.
  // * This method must be invoked on the worker thread.
  virtual void SetMediaChannel(
      cricket::MediaReceiveChannelInterface* media_channel) = 0;

  // Configures the RtpReceiver with the underlying media channel, with the
  // given SSRC as the stream identifier.
  virtual void SetupMediaChannel(uint32_t ssrc) = 0;

  // Configures the RtpReceiver with the underlying media channel to receive an
  // unsignaled receive stream.
  virtual void SetupUnsignaledMediaChannel() = 0;

  virtual void set_transport(
      rtc::scoped_refptr<DtlsTransportInterface> dtls_transport) = 0;
  // This SSRC is used as an identifier for the receiver between the API layer
  // and the WebRtcVideoEngine, WebRtcVoiceEngine layer.
  virtual std::optional<uint32_t> ssrc() const = 0;

  // Call this to notify the RtpReceiver when the first packet has been received
  // on the corresponding channel.
  virtual void NotifyFirstPacketReceived() = 0;

  // Set the associated remote media streams for this receiver. The remote track
  // will be removed from any streams that are no longer present and added to
  // any new streams.
  virtual void set_stream_ids(std::vector<std::string> stream_ids) = 0;
  // TODO(https://crbug.com/webrtc/9480): Remove SetStreams() in favor of
  // set_stream_ids() as soon as downstream projects are no longer dependent on
  // stream objects.
  virtual void SetStreams(
      const std::vector<rtc::scoped_refptr<MediaStreamInterface>>& streams) = 0;

  // Returns an ID that changes if the attached track changes, but
  // otherwise remains constant. Used to generate IDs for stats.
  // The special value zero means that no track is attached.
  virtual int AttachmentId() const = 0;

 protected:
  static int GenerateUniqueId();

  static std::vector<rtc::scoped_refptr<MediaStreamInterface>>
  CreateStreamsFromIds(std::vector<std::string> stream_ids);
};

}  // namespace webrtc

#endif  // PC_RTP_RECEIVER_H_
