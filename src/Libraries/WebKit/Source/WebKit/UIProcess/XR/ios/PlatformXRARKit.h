/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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

#if ENABLE(WEBXR) && USE(ARKITXR_IOS)

#import "PlatformXRCoordinator.h"

#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Threading.h>
#import <wtf/threads/BinarySemaphore.h>

@class ARSession;
@protocol WKARPresentationSession;

namespace WebKit {

class ARKitCoordinator final : public PlatformXRCoordinator {
    WTF_MAKE_TZONE_ALLOCATED(ARKitCoordinator);
    struct RenderState;
public:
    ARKitCoordinator();
    virtual ~ARKitCoordinator() = default;

    void getPrimaryDeviceInfo(WebPageProxy&, DeviceInfoCallback&&) override;
    void requestPermissionOnSessionFeatures(WebPageProxy&, const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList&, const PlatformXR::Device::FeatureList&, const PlatformXR::Device::FeatureList&, const PlatformXR::Device::FeatureList&, const PlatformXR::Device::FeatureList&, FeatureListCallback&&) override;

    void startSession(WebPageProxy&, WeakPtr<SessionEventClient>&&, const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList&) override;
    void endSessionIfExists(WebPageProxy&) override;

    void scheduleAnimationFrame(WebPageProxy&, std::optional<PlatformXR::RequestData>&&, PlatformXR::Device::RequestFrameCallback&&) override;
    void submitFrame(WebPageProxy&) override;

protected:
    void createSessionIfNeeded();
    void endSessionIfExists(std::optional<WebCore::PageIdentifier>);
    void renderLoop(Box<RenderState>);

private:
    XRDeviceIdentifier m_deviceIdentifier = XRDeviceIdentifier::generate();
    RetainPtr<ARSession> m_session;

    struct Idle {
    };
    struct Active {
        WeakPtr<PlatformXRCoordinatorSessionEventClient> sessionEventClient;
        WebCore::PageIdentifier pageIdentifier;
        Box<RenderState> renderState;
        RefPtr<Thread> renderThread;
    };

    using State = std::variant<Idle, Active>;
    State m_state;
};

} // namespace WebKit

#endif // ENABLE(WEBXR) && USE(ARKITXR_IOS)
