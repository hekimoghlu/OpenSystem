/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#include "config.h"
#include "XRDeviceProxy.h"

#if ENABLE(WEBXR)

#include "PlatformXRSystemProxy.h"
#include "XRDeviceInfo.h"
#include <WebCore/SecurityOriginData.h>

using namespace PlatformXR;

namespace WebKit {

Ref<XRDeviceProxy> XRDeviceProxy::create(XRDeviceInfo&& deviceInfo, PlatformXRSystemProxy& xrSystem)
{
    return adoptRef(*new XRDeviceProxy(WTFMove(deviceInfo), xrSystem));
}

XRDeviceProxy::XRDeviceProxy(XRDeviceInfo&& deviceInfo, PlatformXRSystemProxy& xrSystem)
    : m_identifier(deviceInfo.identifier)
    , m_xrSystem(xrSystem)
{
    m_supportsStereoRendering = deviceInfo.supportsStereoRendering;
    m_supportsOrientationTracking = deviceInfo.supportsOrientationTracking;
    m_recommendedResolution = deviceInfo.recommendedResolution;
    m_minimumNearClipPlane = deviceInfo.minimumNearClipPlane;

    if (!deviceInfo.vrFeatures.isEmpty())
        setSupportedFeatures(SessionMode::ImmersiveVr, deviceInfo.vrFeatures);
    if (!deviceInfo.arFeatures.isEmpty())
        setSupportedFeatures(SessionMode::ImmersiveAr, deviceInfo.arFeatures);
}

void XRDeviceProxy::sessionDidEnd()
{
    if (trackingAndRenderingClient())
        trackingAndRenderingClient()->sessionDidEnd();
}

void XRDeviceProxy::updateSessionVisibilityState(PlatformXR::VisibilityState visibilityState)
{
    if (trackingAndRenderingClient())
        trackingAndRenderingClient()->updateSessionVisibilityState(visibilityState);
}

void XRDeviceProxy::initializeTrackingAndRendering(const WebCore::SecurityOriginData& securityOriginData, PlatformXR::SessionMode sessionMode, const PlatformXR::Device::FeatureList& requestedFeatures)
{
    if (!isImmersive(sessionMode))
        return;

    RefPtr xrSystem = m_xrSystem.get();
    if (!xrSystem)
        return;

    xrSystem->initializeTrackingAndRendering();

    // This is called from the constructor of WebXRSession. Since sessionDidInitializeInputSources()
    // ends up calling queueTaskKeepingObjectAlive() which refs the WebXRSession object, we
    // should delay this call after the WebXRSession has finished construction.
    callOnMainRunLoop([this, weakThis = ThreadSafeWeakPtr { *this }]() {
        auto protectedThis = weakThis.get();
        if (!protectedThis)
            return;

        if (trackingAndRenderingClient())
            trackingAndRenderingClient()->sessionDidInitializeInputSources({ });
    });    
}

void XRDeviceProxy::shutDownTrackingAndRendering()
{
    if (RefPtr xrSystem = m_xrSystem.get())
        xrSystem->shutDownTrackingAndRendering();
}

void XRDeviceProxy::didCompleteShutdownTriggeredBySystem()
{
    if (RefPtr xrSystem = m_xrSystem.get())
        xrSystem->didCompleteShutdownTriggeredBySystem();
}

Vector<PlatformXR::Device::ViewData> XRDeviceProxy::views(SessionMode mode) const
{
    Vector<Device::ViewData> views;
    if (m_supportsStereoRendering && mode == SessionMode::ImmersiveVr) {
        views.append({ .active = true, .eye = Eye::Left });
        views.append({ .active = true, .eye = Eye::Right });
    } else
        views.append({ .active = true, .eye = Eye::None });
    return views;
}

void XRDeviceProxy::requestFrame(std::optional<PlatformXR::RequestData>&& requestData, PlatformXR::Device::RequestFrameCallback&& callback)
{
    if (RefPtr xrSystem = m_xrSystem.get())
        xrSystem->requestFrame(WTFMove(requestData), WTFMove(callback));
    else
        callback({ });
}

std::optional<PlatformXR::LayerHandle> XRDeviceProxy::createLayerProjection(uint32_t width, uint32_t height, bool alpha)
{
    RefPtr xrSystem = m_xrSystem.get();
    return xrSystem ? xrSystem->createLayerProjection(width, height, alpha) : std::nullopt;
}

void XRDeviceProxy::submitFrame(Vector<PlatformXR::Device::Layer>&&)
{
    if (RefPtr xrSystem = m_xrSystem.get())
        xrSystem->submitFrame();
}

} // namespace WebKit

#endif // ENABLE(WEBXR)
