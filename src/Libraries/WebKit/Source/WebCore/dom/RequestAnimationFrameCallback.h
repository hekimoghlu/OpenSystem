/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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

#include "ActiveDOMCallback.h"
#include "CallbackResult.h"
#include "DOMHighResTimeStamp.h"
#include <wtf/RefCounted.h>
#include <wtf/Seconds.h>

namespace WebCore {

class RequestAnimationFrameCallback : public RefCounted<RequestAnimationFrameCallback>, public ActiveDOMCallback {
public:
    using ActiveDOMCallback::ActiveDOMCallback;

    virtual CallbackResult<void> handleEvent(double highResTimeMs) = 0;
    virtual CallbackResult<void> handleEventRethrowingException(double highResTimeMs) = 0;

    int m_id;
    bool m_firedOrCancelled;

    // Allow a little more than 60fps to make sure we can at least hit that frame rate.
    static constexpr Seconds fullSpeedAnimationInterval { 15_ms };
    // Allow a little more than 30fps to make sure we can at least hit that frame rate.
    static constexpr Seconds halfSpeedThrottlingAnimationInterval { 30_ms };
    static constexpr Seconds aggressiveThrottlingAnimationInterval { 10_s };

private:
    virtual bool hasCallback() const = 0;
};

} // namespace WebCore
