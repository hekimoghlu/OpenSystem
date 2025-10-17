/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

#include "XRDeviceIdentifier.h"
#include <WebCore/PlatformXR.h>
#include <wtf/Ref.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {

class PlatformXRSystemProxy;

struct XRDeviceInfo;

class XRDeviceProxy final : public PlatformXR::Device {
public:
    static Ref<XRDeviceProxy> create(XRDeviceInfo&&, PlatformXRSystemProxy&);
    XRDeviceIdentifier identifier() const { return m_identifier; }

    void sessionDidEnd();
    void updateSessionVisibilityState(PlatformXR::VisibilityState);

private:
    XRDeviceProxy(XRDeviceInfo&&, PlatformXRSystemProxy&);

    WebCore::IntSize recommendedResolution(PlatformXR::SessionMode) final { return m_recommendedResolution; }
    double minimumNearClipPlane() const final { return m_minimumNearClipPlane; }
    void initializeTrackingAndRendering(const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList&) final;
    void shutDownTrackingAndRendering() final;
    void didCompleteShutdownTriggeredBySystem() final;
    bool supportsSessionShutdownNotification() const final { return true; }
    void initializeReferenceSpace(PlatformXR::ReferenceSpaceType) final { }
    Vector<PlatformXR::Device::ViewData> views(PlatformXR::SessionMode) const final;
    void requestFrame(std::optional<PlatformXR::RequestData>&&, PlatformXR::Device::RequestFrameCallback&&) final;
    std::optional<PlatformXR::LayerHandle> createLayerProjection(uint32_t, uint32_t, bool) final;
    void deleteLayer(PlatformXR::LayerHandle) override { };
    void submitFrame(Vector<PlatformXR::Device::Layer>&&) final;

    XRDeviceIdentifier m_identifier;
    WeakPtr<PlatformXRSystemProxy> m_xrSystem;
    bool m_supportsStereoRendering { false };
    WebCore::IntSize m_recommendedResolution { 0, 0 };
    double m_minimumNearClipPlane { 0.1 };
};

} // namespace WebKit

#endif // ENABLE(WEBXR)
