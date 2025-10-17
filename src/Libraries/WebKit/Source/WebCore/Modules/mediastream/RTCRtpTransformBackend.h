/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class RTCRtpTransformableFrame;

class RTCRtpTransformBackend : public ThreadSafeRefCounted<RTCRtpTransformBackend, WTF::DestructionThread::Main> {
public:
    virtual ~RTCRtpTransformBackend() = default;

    using Callback = Function<void(Ref<RTCRtpTransformableFrame>&&)>;
    virtual void setTransformableFrameCallback(Callback&&) = 0;
    virtual void clearTransformableFrameCallback() = 0;
    virtual void processTransformedFrame(RTCRtpTransformableFrame&) = 0;

    enum class MediaType { Audio, Video };
    virtual MediaType mediaType() const = 0;

    enum class Side { Receiver, Sender };
    virtual Side side() const = 0;

    virtual void requestKeyFrame() = 0;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
