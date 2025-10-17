/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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

#include "MessageReceiver.h"
#include "XRDeviceIdentifier.h"
#include "XRDeviceProxy.h"
#include <WebCore/PlatformXR.h>
#include <wtf/FastMalloc.h>

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {

class WebPage;

class PlatformXRSystemProxy : public IPC::MessageReceiver {
    WTF_MAKE_FAST_ALLOCATED;
public:
    PlatformXRSystemProxy(WebPage&);
    virtual ~PlatformXRSystemProxy();

    void enumerateImmersiveXRDevices(CompletionHandler<void(const PlatformXR::Instance::DeviceList&)>&&);
    void requestPermissionOnSessionFeatures(const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList& /* granted */, const PlatformXR::Device::FeatureList& /* consentRequired */, const PlatformXR::Device::FeatureList& /* consentOptional */, const PlatformXR::Device::FeatureList& /* requiredFeaturesRequested */, const PlatformXR::Device::FeatureList& /* optionalFeaturesRequested */,  CompletionHandler<void(std::optional<PlatformXR::Device::FeatureList>&&)>&&);
    void initializeTrackingAndRendering();
    void shutDownTrackingAndRendering();
    void didCompleteShutdownTriggeredBySystem();
    void requestFrame(std::optional<PlatformXR::RequestData>&&, PlatformXR::Device::RequestFrameCallback&&);
    std::optional<PlatformXR::LayerHandle> createLayerProjection(uint32_t, uint32_t, bool);
    void submitFrame();

    void ref() const final;
    void deref() const final;

private:
    RefPtr<XRDeviceProxy> deviceByIdentifier(XRDeviceIdentifier);
    bool webXREnabled() const;

    Ref<WebPage> protectedPage() const;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Message handlers
    void sessionDidEnd(XRDeviceIdentifier);
    void sessionDidUpdateVisibilityState(XRDeviceIdentifier, PlatformXR::VisibilityState);

    PlatformXR::Instance::DeviceList m_devices;
    WeakRef<WebPage> m_page;
};

}

#endif // ENABLE(WEBXR)
