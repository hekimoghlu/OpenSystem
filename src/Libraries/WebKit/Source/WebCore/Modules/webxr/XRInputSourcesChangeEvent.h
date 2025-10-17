/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#include <wtf/Ref.h>
#include <wtf/Vector.h>

namespace WebCore {

class WebXRInputSource;
class WebXRSession;

class XRInputSourcesChangeEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRInputSourcesChangeEvent);
public:
    struct Init : EventInit {
        RefPtr<WebXRSession> session;
        Vector<Ref<WebXRInputSource>> added;
        Vector<Ref<WebXRInputSource>> removed;
    };

    static Ref<XRInputSourcesChangeEvent> create(const AtomString&, const Init&, IsTrusted = IsTrusted::No);
    virtual ~XRInputSourcesChangeEvent();

    const WebXRSession& session() const;
    const Vector<Ref<WebXRInputSource>>& added() const;
    const Vector<Ref<WebXRInputSource>>& removed() const;

private:
    XRInputSourcesChangeEvent(const AtomString&, const Init&, IsTrusted);

    Ref<WebXRSession> m_session;
    Vector<Ref<WebXRInputSource>> m_added;
    Vector<Ref<WebXRInputSource>> m_removed;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
