/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include "PlatformXRCoordinator.h"
#include "ProcessThrottler.h"
#include <WebCore/PlatformXR.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {

class PlatformXRCoordinator;
class WebPageProxy;

struct SharedPreferencesForWebProcess;
struct XRDeviceInfo;

class PlatformXRSystem : public IPC::MessageReceiver, public PlatformXRCoordinatorSessionEventClient, public RefCounted<PlatformXRSystem> {
    WTF_MAKE_TZONE_ALLOCATED(PlatformXRSystem);
public:
    static Ref<PlatformXRSystem> create(WebPageProxy& page)
    {
        return adoptRef(*new PlatformXRSystem(page));
    }

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    virtual ~PlatformXRSystem();

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    USING_CAN_MAKE_WEAKPTR(PlatformXRCoordinatorSessionEventClient);

    void invalidate();

    bool hasActiveSession() const { return !!m_immersiveSessionActivity; }
    void ensureImmersiveSessionActivity();

private:
    explicit PlatformXRSystem(WebPageProxy&);

    static PlatformXRCoordinator* xrCoordinator();

    bool webXREnabled() const;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Message handlers
    void enumerateImmersiveXRDevices(CompletionHandler<void(Vector<XRDeviceInfo>&&)>&&);
    void requestPermissionOnSessionFeatures(IPC::Connection&, const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList&, const PlatformXR::Device::FeatureList&, const PlatformXR::Device::FeatureList&, const PlatformXR::Device::FeatureList&, const PlatformXR::Device::FeatureList&, CompletionHandler<void(std::optional<PlatformXR::Device::FeatureList>&&)>&&);
    void initializeTrackingAndRendering(IPC::Connection&);
    void shutDownTrackingAndRendering(IPC::Connection&);
    void requestFrame(IPC::Connection&, std::optional<PlatformXR::RequestData>&&, CompletionHandler<void(PlatformXR::FrameData&&)>&&);
    void submitFrame(IPC::Connection&);
    void didCompleteShutdownTriggeredBySystem(IPC::Connection&);

    // PlatformXRCoordinatorSessionEventClient
    void sessionDidEnd(XRDeviceIdentifier) final;
    void sessionDidUpdateVisibilityState(XRDeviceIdentifier, PlatformXR::VisibilityState) final;

    std::optional<PlatformXR::SessionMode> m_immersiveSessionMode;
    std::optional<WebCore::SecurityOriginData> m_immersiveSessionSecurityOriginData;
    std::optional<PlatformXR::Device::FeatureList> m_immersiveSessionGrantedFeatures;
    enum class ImmersiveSessionState : uint8_t {
        Idle,
        RequestingPermissions,
        PermissionsGranted,
        SessionRunning,
        SessionEndingFromWebContent,
        SessionEndingFromSystem,
    };
    ImmersiveSessionState m_immersiveSessionState { ImmersiveSessionState::Idle };
    void setImmersiveSessionState(ImmersiveSessionState, CompletionHandler<void(bool)>&&);
    void invalidateImmersiveSessionState(ImmersiveSessionState nextSessionState = ImmersiveSessionState::Idle);

    WeakPtr<WebPageProxy> m_page;
    RefPtr<ProcessThrottler::ForegroundActivity> m_immersiveSessionActivity;
};

} // namespace WebKit

#endif // ENABLE(WEBXR)
