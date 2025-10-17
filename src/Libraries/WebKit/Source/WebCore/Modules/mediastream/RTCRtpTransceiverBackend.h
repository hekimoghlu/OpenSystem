/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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

#include "ExceptionOr.h"
#include "RTCRtpTransceiverDirection.h"
#include <wtf/Forward.h>

namespace WebCore {
struct RTCRtpCodecCapability;

class RTCRtpTransceiverBackend {
public:
    virtual ~RTCRtpTransceiverBackend() = default;

    virtual RTCRtpTransceiverDirection direction() const = 0;
    virtual std::optional<RTCRtpTransceiverDirection> currentDirection() const = 0;
    virtual void setDirection(RTCRtpTransceiverDirection) = 0;

    virtual String mid() = 0;
    virtual void stop() = 0;
    virtual bool stopped() const = 0;
    virtual ExceptionOr<void> setCodecPreferences(const Vector<RTCRtpCodecCapability>&) = 0;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
