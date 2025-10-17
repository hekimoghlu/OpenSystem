/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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
#pragma once

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "ExceptionCode.h"
#include "RTCIceCandidateFields.h"
#include <webrtc/api/media_types.h>
#include <wtf/text/WTFString.h>

namespace cricket {
class Candidate;
}

namespace webrtc {
struct RtpParameters;
struct RtpTransceiverInit;

class RTCError;

enum class DtlsTransportState;
enum class Priority;
class PriorityValue;
enum class RTCErrorType;
enum class RtpTransceiverDirection;
}

namespace WebCore {

class Exception;
class RTCError;

struct RTCRtpParameters;
struct RTCRtpSendParameters;
struct RTCRtpTransceiverInit;

enum class RTCPriorityType : uint8_t;
enum class RTCRtpTransceiverDirection;

RTCRtpParameters toRTCRtpParameters(const webrtc::RtpParameters&);
void updateRTCRtpSendParameters(const RTCRtpSendParameters&, webrtc::RtpParameters&);
RTCRtpSendParameters toRTCRtpSendParameters(const webrtc::RtpParameters&);
webrtc::RtpParameters fromRTCRtpSendParameters(const RTCRtpSendParameters&, const webrtc::RtpParameters& currentParameters);

RTCRtpTransceiverDirection toRTCRtpTransceiverDirection(webrtc::RtpTransceiverDirection);
webrtc::RtpTransceiverDirection fromRTCRtpTransceiverDirection(RTCRtpTransceiverDirection);
webrtc::RtpTransceiverInit fromRtpTransceiverInit(const RTCRtpTransceiverInit&, cricket::MediaType);

ExceptionCode toExceptionCode(webrtc::RTCErrorType);
Exception toException(const webrtc::RTCError&);
RefPtr<RTCError> toRTCError(const webrtc::RTCError&);

RTCPriorityType toRTCPriorityType(webrtc::PriorityValue);
RTCPriorityType toRTCPriorityType(webrtc::Priority);
webrtc::Priority fromRTCPriorityType(RTCPriorityType);

inline String fromStdString(const std::string& value)
{
    return String::fromUTF8(value);
}

RTCIceCandidateFields convertIceCandidate(const cricket::Candidate&);

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
