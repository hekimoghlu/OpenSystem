/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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

#include "EventTarget.h"
#include "JSDOMPromiseDeferredForward.h"
#include "WebFakeXRDevice.h"
#include "XRSessionMode.h"
#include "XRSimulateUserActivationFunction.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class Document;
class WebXRSystem;

class WebXRTest final : public RefCounted<WebXRTest> {
public:
    struct FakeXRDeviceInit {
        bool supportsImmersive { false };
        std::optional<Vector<XRSessionMode>> supportedModes;
        Vector<FakeXRViewInit> views;

        std::optional<Vector<JSC::JSValue>> supportedFeatures;
        std::optional<Vector<JSC::JSValue>> enabledFeatures;

        std::optional<Vector<FakeXRBoundsPoint>> boundsCoordinates;

        std::optional<FakeXRRigidTransformInit> floorOrigin;
        std::optional<FakeXRRigidTransformInit> viewerOrigin;
    };

    static Ref<WebXRTest> create(WeakPtr<WebXRSystem, WeakPtrImplWithEventTargetData>&& system) { return adoptRef(*new WebXRTest(WTFMove(system))); }
    virtual ~WebXRTest();

    using WebFakeXRDevicePromise = DOMPromiseDeferred<IDLInterface<WebFakeXRDevice>>;
    void simulateDeviceConnection(ScriptExecutionContext& state, const FakeXRDeviceInit&, WebFakeXRDevicePromise&&);

    // Simulates a user activation (aka user gesture) for the current scope.
    // The activation is only guaranteed to be valid in the provided function and only applies to WebXR
    // Device API methods.
    void simulateUserActivation(Document&, XRSimulateUserActivationFunction&);

    // Disconnect all fake devices
    void disconnectAllDevices(DOMPromiseDeferred<void>&&);

private:
    WebXRTest(WeakPtr<WebXRSystem, WeakPtrImplWithEventTargetData>&&);

    WeakPtr<WebXRSystem, WeakPtrImplWithEventTargetData> m_context;
    Vector<Ref<WebFakeXRDevice>> m_devices;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
