/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
// This file contains interfaces for RtpReceivers
// http://w3c.github.io/webrtc-pc/#rtcrtpreceiver-interface

#ifndef API_RTP_RECEIVER_INTERFACE_H_
#define API_RTP_RECEIVER_INTERFACE_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "api/crypto/frame_decryptor_interface.h"
#include "api/dtls_transport_interface.h"
#include "api/frame_transformer_interface.h"
#include "api/media_stream_interface.h"
#include "api/media_types.h"
#include "api/ref_count.h"
#include "api/rtp_parameters.h"
#include "api/scoped_refptr.h"
#include "api/transport/rtp/rtp_source.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class RtpReceiverObserverInterface {
 public:
  // Note: Currently if there are multiple RtpReceivers of the same media type,
  // they will all call OnFirstPacketReceived at once.
  //
  // In the future, it's likely that an RtpReceiver will only call
  // OnFirstPacketReceived when a packet is received specifically for its
  // SSRC/mid.
  virtual void OnFirstPacketReceived(cricket::MediaType media_type) = 0;

 protected:
  virtual ~RtpReceiverObserverInterface() {}
};

class RTC_EXPORT RtpReceiverInterface : public webrtc::RefCountInterface,
                                        public FrameTransformerHost {
 public:
  virtual rtc::scoped_refptr<MediaStreamTrackInterface> track() const = 0;

  // The dtlsTransport attribute exposes the DTLS transport on which the
  // media is received. It may be null.
  // https://w3c.github.io/webrtc-pc/#dom-rtcrtpreceiver-transport
  // TODO(https://bugs.webrtc.org/907849) remove default implementation
  virtual rtc::scoped_refptr<DtlsTransportInterface> dtls_transport() const;

  // The list of streams that `track` is associated with. This is the same as
  // the [[AssociatedRemoteMediaStreams]] internal slot in the spec.
  // https://w3c.github.io/webrtc-pc/#dfn-associatedremotemediastreams
  // TODO(hbos): Make pure virtual as soon as Chromium's mock implements this.
  // TODO(https://crbug.com/webrtc/9480): Remove streams() in favor of
  // stream_ids() as soon as downstream projects are no longer dependent on
  // stream objects.
  virtual std::vector<std::string> stream_ids() const;
  virtual std::vector<rtc::scoped_refptr<MediaStreamInterface>> streams() const;

  // Audio or video receiver?
  virtual cricket::MediaType media_type() const = 0;

  // Not to be confused with "mid", this is a field we can temporarily use
  // to uniquely identify a receiver until we implement Unified Plan SDP.
  virtual std::string id() const = 0;

  // The WebRTC specification only defines RTCRtpParameters in terms of senders,
  // but this API also applies them to receivers, similar to ORTC:
  // http://ortc.org/wp-content/uploads/2016/03/ortc.html#rtcrtpparameters*.
  virtual RtpParameters GetParameters() const = 0;
  // TODO(dinosaurav): Delete SetParameters entirely after rolling to Chromium.
  // Currently, doesn't support changing any parameters.
  virtual bool SetParameters(const RtpParameters& /* parameters */) {
    return false;
  }

  // Does not take ownership of observer.
  // Must call SetObserver(nullptr) before the observer is destroyed.
  virtual void SetObserver(RtpReceiverObserverInterface* observer) = 0;

  // Sets the jitter buffer minimum delay until media playout. Actual observed
  // delay may differ depending on the congestion control. `delay_seconds` is a
  // positive value including 0.0 measured in seconds. `nullopt` means default
  // value must be used.
  virtual void SetJitterBufferMinimumDelay(
      std::optional<double> delay_seconds) = 0;

  // TODO(zhihuang): Remove the default implementation once the subclasses
  // implement this. Currently, the only relevant subclass is the
  // content::FakeRtpReceiver in Chromium.
  virtual std::vector<RtpSource> GetSources() const;

  // Sets a user defined frame decryptor that will decrypt the entire frame
  // before it is sent across the network. This will decrypt the entire frame
  // using the user provided decryption mechanism regardless of whether SRTP is
  // enabled or not.
  // TODO(bugs.webrtc.org/12772): Remove.
  virtual void SetFrameDecryptor(
      rtc::scoped_refptr<FrameDecryptorInterface> frame_decryptor);

  // Returns a pointer to the frame decryptor set previously by the
  // user. This can be used to update the state of the object.
  // TODO(bugs.webrtc.org/12772): Remove.
  virtual rtc::scoped_refptr<FrameDecryptorInterface> GetFrameDecryptor() const;

  // Sets a frame transformer between the depacketizer and the decoder to enable
  // client code to transform received frames according to their own processing
  // logic.
  // TODO: bugs.webrtc.org/15929 - add [[deprecated("Use SetFrameTransformer")]]
  // when usage in Chrome is removed
  virtual void SetDepacketizerToDecoderFrameTransformer(
      rtc::scoped_refptr<FrameTransformerInterface> frame_transformer) {
    SetFrameTransformer(std::move(frame_transformer));
  }

#if defined(WEBRTC_WEBKIT_BUILD)
  virtual void GenerateKeyFrame() { }
#endif

  // Default implementation of SetFrameTransformer.
  // TODO: bugs.webrtc.org/15929 - Make pure virtual.
  void SetFrameTransformer(
      rtc::scoped_refptr<FrameTransformerInterface> frame_transformer) override;

 protected:
  ~RtpReceiverInterface() override = default;
};

}  // namespace webrtc

#endif  // API_RTP_RECEIVER_INTERFACE_H_
