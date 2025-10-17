/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#include "api/rtp_receiver_interface.h"

#include <string>
#include <vector>

#include "api/crypto/frame_decryptor_interface.h"
#include "api/dtls_transport_interface.h"
#include "api/frame_transformer_interface.h"
#include "api/media_stream_interface.h"
#include "api/scoped_refptr.h"
#include "api/transport/rtp/rtp_source.h"

namespace webrtc {

std::vector<std::string> RtpReceiverInterface::stream_ids() const {
  return {};
}

std::vector<rtc::scoped_refptr<MediaStreamInterface>>
RtpReceiverInterface::streams() const {
  return {};
}

std::vector<RtpSource> RtpReceiverInterface::GetSources() const {
  return {};
}

void RtpReceiverInterface::SetFrameDecryptor(
    rtc::scoped_refptr<FrameDecryptorInterface> /* frame_decryptor */) {}

rtc::scoped_refptr<FrameDecryptorInterface>
RtpReceiverInterface::GetFrameDecryptor() const {
  return nullptr;
}

rtc::scoped_refptr<DtlsTransportInterface>
RtpReceiverInterface::dtls_transport() const {
  return nullptr;
}

void RtpReceiverInterface::SetFrameTransformer(
    rtc::scoped_refptr<FrameTransformerInterface> /* frame_transformer */) {}

}  // namespace webrtc
