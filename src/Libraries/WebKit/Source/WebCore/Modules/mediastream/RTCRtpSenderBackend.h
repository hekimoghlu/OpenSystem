/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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

#if ENABLE(WEB_RTC)

#include <wtf/FixedVector.h>
#include <wtf/Forward.h>

namespace WebCore {

class MediaStreamTrack;
class RTCDTMFSenderBackend;
class RTCDtlsTransportBackend;
class RTCRtpSender;
class RTCRtpTransformBackend;
class ScriptExecutionContext;

struct RTCRtpSendParameters;

template<typename IDLType> class DOMPromiseDeferred;

class RTCRtpSenderBackend {
public:
    virtual ~RTCRtpSenderBackend() = default;

    virtual bool replaceTrack(RTCRtpSender&, MediaStreamTrack*) = 0;
    virtual RTCRtpSendParameters getParameters() const = 0;
    virtual void setParameters(const RTCRtpSendParameters&, DOMPromiseDeferred<void>&&) = 0;
    virtual std::unique_ptr<RTCDTMFSenderBackend> createDTMFBackend() = 0;
    virtual Ref<RTCRtpTransformBackend> rtcRtpTransformBackend() = 0;
    virtual void setMediaStreamIds(const FixedVector<String>&) = 0;
    virtual std::unique_ptr<RTCDtlsTransportBackend> dtlsTransportBackend() = 0;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
