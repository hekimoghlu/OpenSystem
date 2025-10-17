/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#if ENABLE(WEBXR)

#include "Event.h"
#include <wtf/RefPtr.h>

namespace WebCore {

class WebXRFrame;
class WebXRInputSource;

class XRInputSourceEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRInputSourceEvent);
public:
    struct Init : EventInit {
        RefPtr<WebXRFrame> frame;
        RefPtr<WebXRInputSource> inputSource;
    };

    static Ref<XRInputSourceEvent> create(const AtomString&, const Init&, IsTrusted = IsTrusted::No);
    virtual ~XRInputSourceEvent();

    const WebXRFrame& frame() const;
    const WebXRInputSource& inputSource() const;
    void setFrameActive(bool);

private:
    XRInputSourceEvent(const AtomString&, const Init&, IsTrusted);

    RefPtr<WebXRFrame> m_frame;
    RefPtr<WebXRInputSource> m_inputSource;
    std::optional<int> m_buttonIndex;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
