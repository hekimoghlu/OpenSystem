/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#include <wtf/Function.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RTCDTMFSenderBackend {
public:
    virtual bool canInsertDTMF() = 0;
    virtual void playTone(const char tone, size_t duration, size_t interToneGap) = 0;
    virtual void onTonePlayed(Function<void()>&&) = 0;

    virtual String tones() const = 0;
    virtual size_t duration() const = 0;
    virtual size_t interToneGap() const = 0;

    virtual ~RTCDTMFSenderBackend() = default;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
