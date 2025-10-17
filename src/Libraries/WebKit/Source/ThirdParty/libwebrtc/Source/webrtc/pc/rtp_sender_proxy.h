/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
#ifndef PC_RTP_SENDER_PROXY_H_
#define PC_RTP_SENDER_PROXY_H_

#include <memory>
#include <string>
#include <vector>

#include "api/rtp_sender_interface.h"
#include "pc/proxy.h"

namespace webrtc {

// Define proxy for RtpSenderInterface.
// TODO(deadbeef): Move this to .cc file. What threads methods are called on is
// an implementation detail.
BEGIN_PRIMARY_PROXY_MAP(RtpSender)
PROXY_PRIMARY_THREAD_DESTRUCTOR()
PROXY_METHOD1(bool, SetTrack, MediaStreamTrackInterface*)
PROXY_CONSTMETHOD0(rtc::scoped_refptr<MediaStreamTrackInterface>, track)
PROXY_CONSTMETHOD0(rtc::scoped_refptr<DtlsTransportInterface>, dtls_transport)
PROXY_CONSTMETHOD0(uint32_t, ssrc)
BYPASS_PROXY_CONSTMETHOD0(cricket::MediaType, media_type)
BYPASS_PROXY_CONSTMETHOD0(std::string, id)
PROXY_CONSTMETHOD0(std::vector<std::string>, stream_ids)
PROXY_CONSTMETHOD0(std::vector<RtpEncodingParameters>, init_send_encodings)
PROXY_CONSTMETHOD0(RtpParameters, GetParameters)
PROXY_METHOD1(RTCError, SetParameters, const RtpParameters&)
PROXY_METHOD2(void,
              SetParametersAsync,
              const RtpParameters&,
              SetParametersCallback)
PROXY_CONSTMETHOD0(rtc::scoped_refptr<DtmfSenderInterface>, GetDtmfSender)
PROXY_METHOD1(void,
              SetFrameEncryptor,
              rtc::scoped_refptr<FrameEncryptorInterface>)
PROXY_METHOD1(void, SetObserver, RtpSenderObserverInterface*)
PROXY_CONSTMETHOD0(rtc::scoped_refptr<FrameEncryptorInterface>,
                   GetFrameEncryptor)
PROXY_METHOD1(void, SetStreams, const std::vector<std::string>&)
PROXY_METHOD1(void,
              SetFrameTransformer,
              rtc::scoped_refptr<FrameTransformerInterface>)
PROXY_METHOD1(void,
              SetEncoderSelector,
              std::unique_ptr<VideoEncoderFactory::EncoderSelectorInterface>)

#if defined(WEBRTC_WEBKIT_BUILD)
PROXY_METHOD1(RTCError, GenerateKeyFrame, const std::vector<std::string>&)
#endif

END_PROXY_MAP(RtpSender)

}  // namespace webrtc

#endif  // PC_RTP_SENDER_PROXY_H_
