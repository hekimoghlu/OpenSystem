/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#include "XRDeviceInfo.h"
#include <WebCore/PlatformXR.h>
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/Function.h>

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {

class WebPageProxy;

class PlatformXRCoordinatorSessionEventClient : public AbstractRefCountedAndCanMakeWeakPtr<PlatformXRCoordinatorSessionEventClient> {
public:
    virtual ~PlatformXRCoordinatorSessionEventClient() = default;

    virtual void sessionDidEnd(XRDeviceIdentifier) = 0;
    virtual void sessionDidUpdateVisibilityState(XRDeviceIdentifier, PlatformXR::VisibilityState) = 0;
};

class PlatformXRCoordinator {
public:
    virtual ~PlatformXRCoordinator() = default;

    // FIXME: Temporary and will be fixed later.
    static PlatformXR::LayerHandle defaultLayerHandle() { return 1; }

    using DeviceInfoCallback = Function<void(std::optional<XRDeviceInfo>)>;
    virtual void getPrimaryDeviceInfo(WebPageProxy&, DeviceInfoCallback&&) = 0;

    using FeatureListCallback = CompletionHandler<void(std::optional<PlatformXR::Device::FeatureList>&&)>;
    virtual void requestPermissionOnSessionFeatures(WebPageProxy&, const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList& granted, const PlatformXR::Device::FeatureList& /* consentRequired */, const PlatformXR::Device::FeatureList& /* consentOptional */, const PlatformXR::Device::FeatureList& /* requiredFeaturesRequested */, const PlatformXR::Device::FeatureList& /* optionalFeaturesRequested */, FeatureListCallback&& completionHandler) { completionHandler(granted); }

    // Session creation/termination.
    virtual void startSession(WebPageProxy&, WeakPtr<PlatformXRCoordinatorSessionEventClient>&&, const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList&) = 0;
    virtual void endSessionIfExists(WebPageProxy&) = 0;

    // Session display loop.
    virtual void scheduleAnimationFrame(WebPageProxy&, std::optional<PlatformXR::RequestData>&&, PlatformXR::Device::RequestFrameCallback&&) = 0;
    virtual void submitFrame(WebPageProxy&) { }
};

} // namespace WebKit

#endif // ENABLE(WEBXR)
