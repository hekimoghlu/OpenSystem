/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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

#if ENABLE(WEBXR) && USE(OPENXR)

#include "GLContext.h"
#include "OpenXRLayer.h"
#include "OpenXRUtils.h"
#include "PlatformXR.h"

#include <wtf/HashMap.h>
#include <wtf/WorkQueue.h>

namespace WebCore {
class GraphicsContextGL;
}
namespace PlatformXR {

class OpenXRExtensions;
class OpenXRInput;

// https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#system
// A system represents a collection of related devices in the runtime, often made up of several individual
// hardware components working together to enable XR experiences.
//
// WebXR talks about XR devices, is a physical unit of hardware that can present imagery to the user, so
// there is not direct correspondence between an OpenXR system and a WebXR device because the system API
// is an abstraction for a collection of devices while the WebXR device is mostly one physical unit,
// usually an HMD or a phone/tablet.
//
// It's important also not to try to associate OpenXR system with WebXR's XRSystem because they're totally
// different concepts. The system in OpenXR was defined above as a collection of related devices. In WebXR,
// the XRSystem is basically the entry point for the WebXR API available via the Navigator object.
class OpenXRDevice final : public Device {
public:
    static Ref<OpenXRDevice> create(XrInstance, XrSystemId, Ref<WorkQueue>&&, const OpenXRExtensions&, CompletionHandler<void()>&&);
    virtual ~OpenXRDevice() = default;

private:
    OpenXRDevice(XrInstance, XrSystemId, Ref<WorkQueue>&&, const OpenXRExtensions&);
    void initialize(CompletionHandler<void()>&& callback);

    // PlatformXR::Device
    WebCore::IntSize recommendedResolution(SessionMode) final;
    void initializeTrackingAndRendering(const WebCore::SecurityOriginData&, SessionMode, const Device::FeatureList&) final;
    void shutDownTrackingAndRendering() final;
    void initializeReferenceSpace(PlatformXR::ReferenceSpaceType) final;
    bool supportsSessionShutdownNotification() const final { return true; }
    void requestFrame(std::optional<RequestData>&&, RequestFrameCallback&&) final;
    void submitFrame(Vector<Device::Layer>&&) final;
    Vector<ViewData> views(SessionMode) const final;
    std::optional<LayerHandle> createLayerProjection(uint32_t width, uint32_t height, bool alpha) final;
    void deleteLayer(LayerHandle) final;

    // Custom methods
    FeatureList collectSupportedFeatures() const;
    void collectSupportedSessionModes();
    void collectConfigurationViews();
    XrSpace createReferenceSpace(XrReferenceSpaceType);
    void pollEvents();
    XrResult beginSession();
    void endSession();
    void resetSession();
    void handleSessionStateChange();
    void waitUntilStopping();
    void updateStageParameters();
    void updateInteractionProfile();
    LayerHandle generateLayerHandle() { return ++m_handleIndex; }

    XrInstance m_instance;
    XrSystemId m_systemId;
    WorkQueue& m_queue;
    const OpenXRExtensions& m_extensions;
    XrSession m_session { XR_NULL_HANDLE };
    XrSessionState m_sessionState { XR_SESSION_STATE_UNKNOWN };
    XrGraphicsBindingEGLMNDX m_graphicsBinding;
    std::unique_ptr<WebCore::GLContext> m_egl;
    RefPtr<WebCore::GraphicsContextGL> m_gl;
    XrFrameState m_frameState;
    Vector<XrView> m_frameViews;
    HashMap<LayerHandle, std::unique_ptr<OpenXRLayer>> m_layers;
    LayerHandle m_handleIndex { 0 };
    std::unique_ptr<OpenXRInput> m_input;
    bool didNotifyInputInitialization { false };

    using ViewConfigurationPropertiesMap = HashMap<XrViewConfigurationType, XrViewConfigurationProperties, IntHash<XrViewConfigurationType>, WTF::StrongEnumHashTraits<XrViewConfigurationType>>;
    ViewConfigurationPropertiesMap m_viewConfigurationProperties;
    using ViewConfigurationViewsMap = HashMap<XrViewConfigurationType, Vector<XrViewConfigurationView>, IntHash<XrViewConfigurationType>, WTF::StrongEnumHashTraits<XrViewConfigurationType>>;
    ViewConfigurationViewsMap m_configurationViews;
    XrViewConfigurationType m_currentViewConfigurationType;
    XrSpace m_localSpace { XR_NULL_HANDLE };
    XrSpace m_viewSpace { XR_NULL_HANDLE };
    XrSpace m_stageSpace { XR_NULL_HANDLE };
    FrameData::StageParameters m_stageParameters;
};

} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
