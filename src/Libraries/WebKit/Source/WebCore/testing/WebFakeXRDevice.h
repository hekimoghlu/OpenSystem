/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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

#include "ExceptionOr.h"
#include "FakeXRBoundsPoint.h"
#include "FakeXRInputSourceInit.h"
#include "FakeXRViewInit.h"
#include "IntSizeHash.h"
#include "JSDOMPromiseDeferredForward.h"
#include "PlatformXR.h"
#include "Timer.h"
#include "WebFakeXRInputController.h"
#include "XRVisibilityState.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {
class GraphicsContextGL;

class FakeXRView final : public RefCounted<FakeXRView> {
public:
    static Ref<FakeXRView> create(XREye eye) { return adoptRef(*new FakeXRView(eye)); }
    using Pose = PlatformXR::FrameData::Pose;
    using Fov = PlatformXR::FrameData::Fov;

    XREye eye() const { return m_eye; }
    const Pose& offset() const { return m_offset; }
    const std::array<float, 16>& projection() const { return m_projection; }
    const std::optional<Fov>& fieldOfView() const { return m_fov;}

    void setResolution(FakeXRViewInit::DeviceResolution resolution) { m_resolution = resolution; }
    void setOffset(Pose&& offset) { m_offset = WTFMove(offset); }
    void setProjection(const Vector<float>&);
    void setFieldOfView(const FakeXRViewInit::FieldOfViewInit&);
private:
    FakeXRView(XREye eye)
        : m_eye(eye) { }


    XREye m_eye;
    FakeXRViewInit::DeviceResolution m_resolution;
    Pose m_offset;
    std::array<float, 16> m_projection;
    std::optional<Fov> m_fov;
};

class SimulatedXRDevice final : public PlatformXR::Device {
    WTF_MAKE_TZONE_ALLOCATED(SimulatedXRDevice);
public:
    SimulatedXRDevice();
    virtual ~SimulatedXRDevice();
    void setViews(Vector<PlatformXR::FrameData::View>&&);
    void setNativeBoundsGeometry(const Vector<FakeXRBoundsPoint>&);
    void setViewerOrigin(const std::optional<PlatformXR::FrameData::Pose>&);
    void setFloorOrigin(std::optional<PlatformXR::FrameData::Pose>&& origin) { m_frameData.floorTransform = WTFMove(origin); }
    void setEmulatedPosition(bool emulated) { m_frameData.isPositionEmulated = emulated; }
    void setSupportsShutdownNotification(bool supportsShutdownNotification) { m_supportsShutdownNotification = supportsShutdownNotification; }
    void setVisibilityState(XRVisibilityState);
    void simulateShutdownCompleted();
    void scheduleOnNextFrame(Function<void()>&&);
    void addInputConnection(Ref<WebFakeXRInputController>&& input) { m_inputConnections.append(WTFMove(input)); };
private:
    WebCore::IntSize recommendedResolution(PlatformXR::SessionMode) final;
    void initializeTrackingAndRendering(const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList&) final;
    void shutDownTrackingAndRendering() final;
    bool supportsSessionShutdownNotification() const final { return m_supportsShutdownNotification; }
    void initializeReferenceSpace(PlatformXR::ReferenceSpaceType) final { }
    Vector<PlatformXR::Device::ViewData> views(PlatformXR::SessionMode) const final;
    void requestFrame(std::optional<PlatformXR::RequestData>&&, RequestFrameCallback&&) final;
    std::optional<PlatformXR::LayerHandle> createLayerProjection(uint32_t width, uint32_t height, bool alpha) final;
    void deleteLayer(PlatformXR::LayerHandle) final;

    void stopTimer();
    void frameTimerFired();

    PlatformXR::FrameData m_frameData;
    bool m_supportsShutdownNotification { false };
    Timer m_frameTimer;
    RequestFrameCallback m_FrameCallback;
#if PLATFORM(COCOA)
    HashMap<PlatformXR::LayerHandle, WebCore::IntSize> m_layers;
#else
    HashMap<PlatformXR::LayerHandle, PlatformGLObject> m_layers;
    RefPtr<WebCore::GraphicsContextGL> m_gl;
#endif
    uint32_t m_layerIndex { 0 };
    Vector<Ref<WebFakeXRInputController>> m_inputConnections;
};

class WebFakeXRDevice final : public RefCounted<WebFakeXRDevice> {
public:
    static Ref<WebFakeXRDevice> create() { return adoptRef(*new WebFakeXRDevice()); }

    void setViews(const Vector<FakeXRViewInit>&);
    void disconnect(DOMPromiseDeferred<void>&&);
    void setViewerOrigin(FakeXRRigidTransformInit origin, bool emulatedPosition = false);
    void clearViewerOrigin() { m_device->setViewerOrigin(std::nullopt); }
    void simulateVisibilityChange(XRVisibilityState);
    void setBoundsGeometry(Vector<FakeXRBoundsPoint>&& bounds) { m_device->setNativeBoundsGeometry(WTFMove(bounds)); }
    void setFloorOrigin(FakeXRRigidTransformInit);
    void clearFloorOrigin() { m_device->setFloorOrigin(std::nullopt); }
    void simulateResetPose();
    Ref<WebFakeXRInputController> simulateInputSourceConnection(const FakeXRInputSourceInit&);
    static ExceptionOr<Ref<FakeXRView>> parseView(const FakeXRViewInit&);
    SimulatedXRDevice& simulatedXRDevice() { return m_device; }
    void setSupportsShutdownNotification();
    void simulateShutdown();

    static ExceptionOr<PlatformXR::FrameData::Pose> parseRigidTransform(const FakeXRRigidTransformInit&);

private:
    WebFakeXRDevice();

    Ref<SimulatedXRDevice> m_device;
    PlatformXR::InputSourceHandle mInputSourceHandleIndex { 0 };
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
