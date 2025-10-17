/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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

#include "RTCRtpSFrameTransform.h"
#include "RTCRtpScriptTransform.h"
#include "RTCRtpTransformBackend.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RTCRtpReceiver;
class RTCRtpSender;

class RTCRtpTransform  {
    WTF_MAKE_TZONE_ALLOCATED(RTCRtpTransform);
public:
    using Internal = std::variant<RefPtr<RTCRtpSFrameTransform>, RefPtr<RTCRtpScriptTransform>>;
    static std::unique_ptr<RTCRtpTransform> from(std::optional<Internal>&&);

    explicit RTCRtpTransform(Internal&&);
    ~RTCRtpTransform();

    bool isAttached() const;
    void attachToReceiver(RTCRtpReceiver&, RTCRtpTransform*);
    void attachToSender(RTCRtpSender&, RTCRtpTransform*);
    void detachFromReceiver(RTCRtpReceiver&);
    void detachFromSender(RTCRtpSender&);

    RefPtr<RTCRtpTransformBackend> takeBackend() { return WTFMove(m_backend); }
    Internal internalTransform() { return m_transform; }

    friend bool operator==(const RTCRtpTransform&, const RTCRtpTransform&);

private:
    void clearBackend();
    void backendTransferedToNewTransform();

    RefPtr<RTCRtpTransformBackend> m_backend;
    Internal m_transform;
};

bool operator==(const RTCRtpTransform&, const RTCRtpTransform&);

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
