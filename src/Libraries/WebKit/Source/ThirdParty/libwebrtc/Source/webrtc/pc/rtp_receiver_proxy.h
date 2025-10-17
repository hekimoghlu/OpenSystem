/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#ifndef PC_RTP_RECEIVER_PROXY_H_
#define PC_RTP_RECEIVER_PROXY_H_

#include <optional>
#include <string>
#include <vector>

#include "api/crypto/frame_decryptor_interface.h"
#include "api/dtls_transport_interface.h"
#include "api/frame_transformer_interface.h"
#include "api/media_stream_interface.h"
#include "api/media_types.h"
#include "api/rtp_parameters.h"
#include "api/rtp_receiver_interface.h"
#include "api/scoped_refptr.h"
#include "api/transport/rtp/rtp_source.h"
#include "pc/proxy.h"

namespace webrtc {

// Define proxy for RtpReceiverInterface.
// TODO(deadbeef): Move this to .cc file. What threads methods are called on is
// an implementation detail.
BEGIN_PROXY_MAP(RtpReceiver)
PROXY_PRIMARY_THREAD_DESTRUCTOR()
BYPASS_PROXY_CONSTMETHOD0(rtc::scoped_refptr<MediaStreamTrackInterface>, track)
PROXY_CONSTMETHOD0(rtc::scoped_refptr<DtlsTransportInterface>, dtls_transport)
PROXY_CONSTMETHOD0(std::vector<std::string>, stream_ids)
PROXY_CONSTMETHOD0(std::vector<rtc::scoped_refptr<MediaStreamInterface>>,
                   streams)
BYPASS_PROXY_CONSTMETHOD0(cricket::MediaType, media_type)
BYPASS_PROXY_CONSTMETHOD0(std::string, id)
PROXY_SECONDARY_CONSTMETHOD0(RtpParameters, GetParameters)
PROXY_METHOD1(void, SetObserver, RtpReceiverObserverInterface*)
PROXY_SECONDARY_METHOD1(void,
                        SetJitterBufferMinimumDelay,
                        std::optional<double>)
PROXY_SECONDARY_CONSTMETHOD0(std::vector<RtpSource>, GetSources)
// TODO(bugs.webrtc.org/12772): Remove.
PROXY_SECONDARY_METHOD1(void,
                        SetFrameDecryptor,
                        rtc::scoped_refptr<FrameDecryptorInterface>)
// TODO(bugs.webrtc.org/12772): Remove.
PROXY_SECONDARY_CONSTMETHOD0(rtc::scoped_refptr<FrameDecryptorInterface>,
                             GetFrameDecryptor)
PROXY_SECONDARY_METHOD1(void,
                        SetFrameTransformer,
                        rtc::scoped_refptr<FrameTransformerInterface>)
#if defined(WEBRTC_WEBKIT_BUILD)
PROXY_METHOD0(void, GenerateKeyFrame)
#endif
END_PROXY_MAP(RtpReceiver)

}  // namespace webrtc

#endif  // PC_RTP_RECEIVER_PROXY_H_
