/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "HTMLCanvasElement.h"
#include "JSDOMPromiseDeferredForward.h"
#include "PlatformXR.h"
#include "WebGLContextAttributes.h"
#include "WebGLRenderingContextBase.h"
#include "XRReferenceSpaceType.h"
#include "XRSessionMode.h"
#include <wtf/HashSet.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakHashSet.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace JSC {
class JSGlobalObject;
}

namespace WebCore {

class LocalDOMWindow;
class Navigator;
class ScriptExecutionContext;
class WebXRSession;
class SecurityOriginData;
struct XRSessionInit;

class WebXRSystem final : public RefCounted<WebXRSystem>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRSystem);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    using IsSessionSupportedPromise = DOMPromiseDeferred<IDLBoolean>;
    using RequestSessionPromise = DOMPromiseDeferred<IDLInterface<WebXRSession>>;

    static Ref<WebXRSystem> create(Navigator&);
    ~WebXRSystem();

    void isSessionSupported(XRSessionMode, IsSessionSupportedPromise&&);
    void requestSession(Document&, XRSessionMode, const XRSessionInit&, RequestSessionPromise&&);

    // This is also needed by WebGLRenderingContextBase::makeXRCompatible() and HTMLCanvasElement::createContextWebGL().
    void ensureImmersiveXRDeviceIsSelected(CompletionHandler<void()>&&);
    bool hasActiveImmersiveXRDevice() const { return !!m_activeImmersiveDevice.get(); }

    RefPtr<WebXRSession> activeImmersiveSession() const;
    void sessionEnded(WebXRSession&);

    // For testing purpouses only.
    WEBCORE_EXPORT void registerSimulatedXRDeviceForTesting(PlatformXR::Device&);
    WEBCORE_EXPORT void unregisterSimulatedXRDeviceForTesting(PlatformXR::Device&);

    Navigator* navigator();

protected:
    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const override { return EventTargetInterfaceType::WebXRSystem; }
    ScriptExecutionContext* scriptExecutionContext() const override { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() override { ref(); }
    void derefEventTarget() override { deref(); }

    // ActiveDOMObject
    void stop() override;

private:
    WebXRSystem(Navigator&);

    using FeatureList = PlatformXR::Device::FeatureList;
    using JSFeatureList = Vector<JSC::JSValue>;
    void obtainCurrentDevice(XRSessionMode, const JSFeatureList& requiredFeatures, const JSFeatureList& optionalFeatures, CompletionHandler<void(ThreadSafeWeakPtr<PlatformXR::Device>)>&&);

    bool immersiveSessionRequestIsAllowedForGlobalObject(LocalDOMWindow&, Document&) const;
    bool inlineSessionRequestIsAllowedForGlobalObject(LocalDOMWindow&, Document&, const XRSessionInit&) const;

    bool isFeaturePermitted(PlatformXR::SessionFeature) const;
    bool isFeatureSupported(PlatformXR::SessionFeature, XRSessionMode, const PlatformXR::Device&) const;
    struct ResolvedRequestedFeatures;
    std::optional<ResolvedRequestedFeatures> resolveRequestedFeatures(XRSessionMode, const XRSessionInit&, RefPtr<PlatformXR::Device>, JSC::JSGlobalObject&) const;
    void resolveFeaturePermissions(XRSessionMode, const XRSessionInit&, RefPtr<PlatformXR::Device>, JSC::JSGlobalObject&, CompletionHandler<void(std::optional<FeatureList>&&)>&&) const;

    // https://immersive-web.github.io/webxr/#default-inline-xr-device
    class DummyInlineDevice final : public PlatformXR::Device, public ContextDestructionObserver {
    public:
        static Ref<DummyInlineDevice> create(ScriptExecutionContext&);
        virtual ~DummyInlineDevice() = default;

    private:
        DummyInlineDevice(ScriptExecutionContext&);

        void initializeTrackingAndRendering(const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList&) final { }
        void shutDownTrackingAndRendering() final { }
        void initializeReferenceSpace(PlatformXR::ReferenceSpaceType) final { }

        void requestFrame(std::optional<PlatformXR::RequestData>&&, PlatformXR::Device::RequestFrameCallback&&) final;
        Vector<Device::ViewData> views(XRSessionMode) const final;
        std::optional<PlatformXR::LayerHandle> createLayerProjection(uint32_t, uint32_t, bool) final { return std::nullopt; }
        void deleteLayer(PlatformXR::LayerHandle) final { }
    };

    WeakPtr<Navigator> m_navigator;
    Ref<PlatformXR::Device> m_defaultInlineDevice;

    bool m_immersiveXRDevicesHaveBeenEnumerated { false };
    uint m_testingDevices { 0 };

    bool m_pendingImmersiveSession { false };
    RefPtr<WebXRSession> m_activeImmersiveSession;

    // We use a set here although the specs talk about a list of inline sessions
    // https://immersive-web.github.io/webxr/#list-of-inline-sessions.
    HashSet<Ref<WebXRSession>> m_inlineSessions;

    ThreadSafeWeakPtr<PlatformXR::Device> m_activeImmersiveDevice;
    ThreadSafeWeakHashSet<PlatformXR::Device> m_immersiveDevices;
    ThreadSafeWeakPtr<PlatformXR::Device> m_inlineXRDevice;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
