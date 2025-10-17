/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
#include "PlatformXRSystemProxy.h"

#if ENABLE(WEBXR)

#include "MessageSenderInlines.h"
#include "PlatformXRCoordinator.h"
#include "PlatformXRSystemMessages.h"
#include "PlatformXRSystemProxyMessages.h"
#include "WebPage.h"
#include "WebProcess.h"
#include "XRDeviceInfo.h"
#include <WebCore/Page.h>
#include <WebCore/SecurityOrigin.h>
#include <wtf/Vector.h>

using namespace PlatformXR;

namespace WebKit {

PlatformXRSystemProxy::PlatformXRSystemProxy(WebPage& page)
    : m_page(page)
{
    WebProcess::singleton().addMessageReceiver(Messages::PlatformXRSystemProxy::messageReceiverName(), m_page->identifier(), *this);
}

PlatformXRSystemProxy::~PlatformXRSystemProxy()
{
    WebProcess::singleton().removeMessageReceiver(Messages::PlatformXRSystemProxy::messageReceiverName(), m_page->identifier());
}

Ref<WebPage> PlatformXRSystemProxy::protectedPage() const
{
    return m_page.get();
}

void PlatformXRSystemProxy::enumerateImmersiveXRDevices(CompletionHandler<void(const Instance::DeviceList&)>&& completionHandler)
{
    protectedPage()->sendWithAsyncReply(Messages::PlatformXRSystem::EnumerateImmersiveXRDevices(), [this, weakThis = WeakPtr { *this }, completionHandler = WTFMove(completionHandler)](Vector<XRDeviceInfo>&& devicesInfos) mutable {
        if (!weakThis)
            return;

        PlatformXR::Instance::DeviceList devices;
        for (auto& deviceInfo : devicesInfos) {
            if (auto device = deviceByIdentifier(deviceInfo.identifier))
                devices.append(*device);
            else
                devices.append(XRDeviceProxy::create(WTFMove(deviceInfo), *this));
        }
        m_devices.swap(devices);
        completionHandler(m_devices);
    });
}

void PlatformXRSystemProxy::requestPermissionOnSessionFeatures(const WebCore::SecurityOriginData& securityOriginData, PlatformXR::SessionMode mode, const PlatformXR::Device::FeatureList& granted, const PlatformXR::Device::FeatureList& consentRequired, const PlatformXR::Device::FeatureList& consentOptional, const PlatformXR::Device::FeatureList& requiredFeaturesRequested, const PlatformXR::Device::FeatureList& optionalFeaturesRequested, CompletionHandler<void(std::optional<PlatformXR::Device::FeatureList>&&)>&& completionHandler)
{
    protectedPage()->sendWithAsyncReply(Messages::PlatformXRSystem::RequestPermissionOnSessionFeatures(securityOriginData, mode, granted, consentRequired, consentOptional, requiredFeaturesRequested, optionalFeaturesRequested), WTFMove(completionHandler));
}

void PlatformXRSystemProxy::initializeTrackingAndRendering()
{
    protectedPage()->send(Messages::PlatformXRSystem::InitializeTrackingAndRendering());
}

void PlatformXRSystemProxy::shutDownTrackingAndRendering()
{
    protectedPage()->send(Messages::PlatformXRSystem::ShutDownTrackingAndRendering());
}

void PlatformXRSystemProxy::didCompleteShutdownTriggeredBySystem()
{
    protectedPage()->send(Messages::PlatformXRSystem::DidCompleteShutdownTriggeredBySystem());
}

void PlatformXRSystemProxy::requestFrame(std::optional<PlatformXR::RequestData>&& requestData, PlatformXR::Device::RequestFrameCallback&& callback)
{
    protectedPage()->sendWithAsyncReply(Messages::PlatformXRSystem::RequestFrame(WTFMove(requestData)), WTFMove(callback));
}

std::optional<PlatformXR::LayerHandle> PlatformXRSystemProxy::createLayerProjection(uint32_t, uint32_t, bool)
{
    return PlatformXRCoordinator::defaultLayerHandle();
}

void PlatformXRSystemProxy::submitFrame()
{
    protectedPage()->send(Messages::PlatformXRSystem::SubmitFrame());
}

void PlatformXRSystemProxy::sessionDidEnd(XRDeviceIdentifier deviceIdentifier)
{
    RELEASE_ASSERT(webXREnabled());

    if (auto device = deviceByIdentifier(deviceIdentifier))
        device->sessionDidEnd();
}

void PlatformXRSystemProxy::sessionDidUpdateVisibilityState(XRDeviceIdentifier deviceIdentifier, PlatformXR::VisibilityState visibilityState)
{
    RELEASE_ASSERT(webXREnabled());

    if (auto device = deviceByIdentifier(deviceIdentifier))
        device->updateSessionVisibilityState(visibilityState);
}

RefPtr<XRDeviceProxy> PlatformXRSystemProxy::deviceByIdentifier(XRDeviceIdentifier identifier)
{
    for (auto& device : m_devices) {
        auto* deviceProxy = static_cast<XRDeviceProxy*>(device.ptr());
        if (deviceProxy->identifier() == identifier)
            return deviceProxy;
    }

    return nullptr;
}

bool PlatformXRSystemProxy::webXREnabled() const
{
    Ref page = m_page.get();
    return page->corePage() && page->corePage()->settings().webXREnabled();
}

void PlatformXRSystemProxy::ref() const
{
    m_page->ref();
}

void PlatformXRSystemProxy::deref() const
{
    m_page->deref();
}

} // namespace WebKit

#endif // ENABLE(WEBXR)
