/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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

#if ENABLE(MEDIA_STREAM)

#include "IdentifierTypes.h"
#include "SandboxExtension.h"
#include <WebCore/MediaCanStartListener.h>
#include <WebCore/MediaConstraints.h>
#include <WebCore/RealtimeMediaSourceCenter.h>
#include <WebCore/UserMediaClient.h>
#include <WebCore/UserMediaRequest.h>
#include <wtf/HashMap.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebPage;

class UserMediaPermissionRequestManager : public WebCore::MediaCanStartListener
#if USE(GSTREAMER)
                                        , public WebCore::RealtimeMediaSourceCenterObserver
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(UserMediaPermissionRequestManager);
public:
    USING_CAN_MAKE_WEAKPTR(WebCore::MediaCanStartListener);

    explicit UserMediaPermissionRequestManager(WebPage&);
    ~UserMediaPermissionRequestManager() = default;

    void ref() const final;
    void deref() const final;

    void startUserMediaRequest(WebCore::UserMediaRequest&);
    void cancelUserMediaRequest(WebCore::UserMediaRequest&);
    void userMediaAccessWasGranted(WebCore::UserMediaRequestIdentifier, WebCore::CaptureDevice&& audioDevice, WebCore::CaptureDevice&& videoDevice, WebCore::MediaDeviceHashSalts&&, CompletionHandler<void()>&&);
    void userMediaAccessWasDenied(WebCore::UserMediaRequestIdentifier, WebCore::MediaAccessDenialReason, String&&, WebCore::MediaConstraintType);

    void enumerateMediaDevices(WebCore::Document&, CompletionHandler<void(Vector<WebCore::CaptureDeviceWithCapabilities>&&, WebCore::MediaDeviceHashSalts&&)>&&);

    WebCore::UserMediaClient::DeviceChangeObserverToken addDeviceChangeObserver(WTF::Function<void()>&&);
    void removeDeviceChangeObserver(WebCore::UserMediaClient::DeviceChangeObserverToken);
    void updateCaptureState(const WebCore::Document&, bool isActive, WebCore::MediaProducerMediaCaptureKind, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&&);

    void captureDevicesChanged();

private:
#if USE(GSTREAMER)
    // WebCore::RealtimeMediaSourceCenterObserver
    void devicesChanged() final;
    void deviceWillBeRemoved(const String& persistentId) final { }
#endif

    void sendUserMediaRequest(WebCore::UserMediaRequest&);

    // WebCore::MediaCanStartListener
    void mediaCanStart(WebCore::Document&) final;

    Ref<WebPage> protectedPage() const;

    WeakRef<WebPage> m_page;

    HashMap<WebCore::UserMediaRequestIdentifier, Ref<WebCore::UserMediaRequest>> m_ongoingUserMediaRequests;
    HashMap<RefPtr<WebCore::Document>, Vector<Ref<WebCore::UserMediaRequest>>> m_pendingUserMediaRequests;

    HashMap<WebCore::UserMediaClient::DeviceChangeObserverToken, Function<void()>> m_deviceChangeObserverMap;
    bool m_monitoringDeviceChange { false };

#if USE(GSTREAMER)
    enum class ShouldNotify : bool { No, Yes };
    void updateCaptureDevices(ShouldNotify);

    Vector<WebCore::CaptureDevice> m_captureDevices;
#endif
};

} // namespace WebKit

namespace WTF {

} // namespace WTF

#endif // ENABLE(MEDIA_STREAM)
