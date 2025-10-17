/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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

#include "PlatformXR.h"
#include "WebXRGamepad.h"
#include "WebXRInputSpace.h"
#include "XRHandedness.h"
#include "XRTargetRayMode.h"
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

#if ENABLE(WEBXR_HANDS)
#include "WebXRHand.h"
#endif

namespace WebCore {

#if ENABLE(GAMEPAD)
class Gamepad;
#endif
class XRInputSourceEvent;
class WebXRHand;
class WebXRInputSpace;

class WebXRInputSource : public RefCountedAndCanMakeWeakPtr<WebXRInputSource> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRInputSource);
public:
    using InputSource = PlatformXR::FrameData::InputSource;
    using InputSourceButton = PlatformXR::FrameData::InputSourceButton;

    static Ref<WebXRInputSource> create(Document&, WebXRSession&, double timestamp, const InputSource&);
    ~WebXRInputSource();

    PlatformXR::InputSourceHandle handle() const { return m_source.handle; }
    XRHandedness handedness() const { return m_source.handedness; }
    XRTargetRayMode targetRayMode() const { return m_source.targetRayMode; };
    const WebXRSpace& targetRaySpace() const {return m_targetRaySpace.get(); };
    WebXRSpace* gripSpace() const { return m_gripSpace.get(); }
    const Vector<String>& profiles() const { return m_source.profiles; };
    double connectTime() const { return m_connectTime; }
#if ENABLE(GAMEPAD)
    Gamepad* gamepad() const { return m_gamepad.ptr(); }
#endif

#if ENABLE(WEBXR_HANDS)
    WebXRHand* hand() const { return m_hand.get(); }
#endif

    void update(double timestamp, const InputSource&);
    bool requiresInputSourceChange(const InputSource&);
    void disconnect();

    void pollEvents(Vector<Ref<XRInputSourceEvent>>&);

    // For GC reachablitiy.
    WebXRSession* session();

private:
    WebXRInputSource(Document&, WebXRSession&, double timestamp, const InputSource&);

    WeakPtr<WebXRSession> m_session;
    InputSource m_source;
    Ref<WebXRInputSpace> m_targetRaySpace;
    RefPtr<WebXRInputSpace> m_gripSpace;
    double m_connectTime { 0 };
    bool m_connected { true };
#if ENABLE(GAMEPAD)
    Ref<Gamepad> m_gamepad;
#endif

#if ENABLE(WEBXR_HANDS)
    RefPtr<WebXRHand> m_hand;
#endif

    bool m_selectStarted { false };
    bool m_squeezeStarted { false };
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
