/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#ifndef API_TEST_MOCK_PEER_CONNECTION_FACTORY_INTERFACE_H_
#define API_TEST_MOCK_PEER_CONNECTION_FACTORY_INTERFACE_H_

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "api/audio_options.h"
#include "api/media_stream_interface.h"
#include "api/media_types.h"
#include "api/peer_connection_interface.h"
#include "api/rtc_error.h"
#include "api/rtp_parameters.h"
#include "api/scoped_refptr.h"
#include "p2p/base/port_allocator.h"
#include "rtc_base/ref_counted_object.h"
#include "rtc_base/rtc_certificate_generator.h"
#include "test/gmock.h"

namespace webrtc {

class MockPeerConnectionFactoryInterface
    : public rtc::RefCountedObject<webrtc::PeerConnectionFactoryInterface> {
 public:
  static rtc::scoped_refptr<MockPeerConnectionFactoryInterface> Create() {
    return rtc::scoped_refptr<MockPeerConnectionFactoryInterface>(
        new MockPeerConnectionFactoryInterface());
  }

  MOCK_METHOD(void, SetOptions, (const Options&), (override));
  MOCK_METHOD(rtc::scoped_refptr<PeerConnectionInterface>,
              CreatePeerConnection,
              (const PeerConnectionInterface::RTCConfiguration&,
               PeerConnectionDependencies),
              (override));
  MOCK_METHOD(RTCErrorOr<rtc::scoped_refptr<PeerConnectionInterface>>,
              CreatePeerConnectionOrError,
              (const PeerConnectionInterface::RTCConfiguration&,
               PeerConnectionDependencies),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<PeerConnectionInterface>,
              CreatePeerConnection,
              (const PeerConnectionInterface::RTCConfiguration&,
               std::unique_ptr<cricket::PortAllocator>,
               std::unique_ptr<rtc::RTCCertificateGeneratorInterface>,
               PeerConnectionObserver*),
              (override));
  MOCK_METHOD(RtpCapabilities,
              GetRtpSenderCapabilities,
              (cricket::MediaType),
              (const, override));
  MOCK_METHOD(RtpCapabilities,
              GetRtpReceiverCapabilities,
              (cricket::MediaType),
              (const, override));
  MOCK_METHOD(rtc::scoped_refptr<MediaStreamInterface>,
              CreateLocalMediaStream,
              (const std::string&),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<AudioSourceInterface>,
              CreateAudioSource,
              (const cricket::AudioOptions&),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<VideoTrackInterface>,
              CreateVideoTrack,
              (const std::string&, VideoTrackSourceInterface*),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<VideoTrackInterface>,
              CreateVideoTrack,
              (rtc::scoped_refptr<VideoTrackSourceInterface>,
               absl::string_view),
              (override));
  MOCK_METHOD(rtc::scoped_refptr<AudioTrackInterface>,
              CreateAudioTrack,
              (const std::string&, AudioSourceInterface*),
              (override));
  MOCK_METHOD(bool, StartAecDump, (FILE*, int64_t), (override));
  MOCK_METHOD(void, StopAecDump, (), (override));

 protected:
  MockPeerConnectionFactoryInterface() = default;
};

}  // namespace webrtc

#endif  // API_TEST_MOCK_PEER_CONNECTION_FACTORY_INTERFACE_H_
