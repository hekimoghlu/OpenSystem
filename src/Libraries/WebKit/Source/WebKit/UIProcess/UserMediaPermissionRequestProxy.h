/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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

#include "APIObject.h"
#include <WebCore/CaptureDevice.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/MediaDeviceHashSalts.h>
#include <WebCore/MediaStreamRequest.h>
#include <WebCore/UserMediaRequestIdentifier.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class SecurityOrigin;
}

namespace WebKit {

class UserMediaPermissionRequestManagerProxy;

class UserMediaPermissionRequestProxy : public API::ObjectImpl<API::Object::Type::UserMediaPermissionRequest> {
public:
    static Ref<UserMediaPermissionRequestProxy> create(UserMediaPermissionRequestManagerProxy&, std::optional<WebCore::UserMediaRequestIdentifier>, WebCore::FrameIdentifier, WebCore::FrameIdentifier, Ref<WebCore::SecurityOrigin>&&, Ref<WebCore::SecurityOrigin>&&, Vector<WebCore::CaptureDevice>&&, Vector<WebCore::CaptureDevice>&&, WebCore::MediaStreamRequest&&, CompletionHandler<void(bool)>&& = { });

    ~UserMediaPermissionRequestProxy();

    void allow(const String& audioDeviceUID, const String& videoDeviceUID);
    void allow();
    void promptForGetUserMedia();

    enum class UserMediaDisplayCapturePromptType { Window, Screen, UserChoose };
    virtual void promptForGetDisplayMedia(UserMediaDisplayCapturePromptType);
    virtual bool canRequestDisplayCapturePermission();

    void doDefaultAction();
    enum class UserMediaAccessDenialReason { NoConstraints, UserMediaDisabled, NoCaptureDevices, InvalidConstraint, HardwareError, PermissionDenied, OtherFailure };
    void deny(UserMediaAccessDenialReason = UserMediaAccessDenialReason::UserMediaDisabled);

    virtual void invalidate();
    bool isPending() const;

    bool requiresAudioCapture() const { return m_eligibleAudioDevices.size(); }
    bool requiresVideoCapture() const { return !requiresDisplayCapture() && m_eligibleVideoDevices.size(); }
    bool requiresDisplayCapture() const { return m_request.type == WebCore::MediaStreamRequest::Type::DisplayMedia || m_request.type == WebCore::MediaStreamRequest::Type::DisplayMediaWithAudio; }
    bool requiresDisplayCaptureWithAudio() const { return m_request.type == WebCore::MediaStreamRequest::Type::DisplayMediaWithAudio; }

    void setEligibleVideoDeviceUIDs(Vector<WebCore::CaptureDevice>&& devices) { m_eligibleVideoDevices = WTFMove(devices); }
    void setEligibleAudioDeviceUIDs(Vector<WebCore::CaptureDevice>&& devices) { m_eligibleAudioDevices = WTFMove(devices); }

    Vector<String> videoDeviceUIDs() const;
    Vector<String> audioDeviceUIDs() const;
    bool hasAudioDevice() const { return !m_eligibleAudioDevices.isEmpty(); }
    bool hasVideoDevice() const { return !m_eligibleVideoDevices.isEmpty(); }

    bool hasPersistentAccess() const { return m_hasPersistentAccess; }
    void setHasPersistentAccess() { m_hasPersistentAccess = true; }

    std::optional<WebCore::UserMediaRequestIdentifier> userMediaID() const { return m_userMediaID; }
    WebCore::FrameIdentifier mainFrameID() const { return m_mainFrameID; }
    WebCore::FrameIdentifier frameID() const { return m_frameID; }

    WebCore::SecurityOrigin& topLevelDocumentSecurityOrigin() { return m_topLevelDocumentSecurityOrigin.get(); }
    WebCore::SecurityOrigin& userMediaDocumentSecurityOrigin() { return m_userMediaDocumentSecurityOrigin.get(); }
    const WebCore::SecurityOrigin& topLevelDocumentSecurityOrigin() const { return m_topLevelDocumentSecurityOrigin.get(); }
    const WebCore::SecurityOrigin& userMediaDocumentSecurityOrigin() const { return m_userMediaDocumentSecurityOrigin.get(); }
    Ref<WebCore::SecurityOrigin> protectedTopLevelDocumentSecurityOrigin() const { return m_topLevelDocumentSecurityOrigin.get(); }
    Ref<WebCore::SecurityOrigin> protectedUserMediaDocumentSecurityOrigin() const { return m_userMediaDocumentSecurityOrigin.get(); }

    const WebCore::MediaStreamRequest& userRequest() const { return m_request; }

    WebCore::MediaStreamRequest::Type requestType() const { return m_request.type; }

    void setDeviceIdentifierHashSalts(WebCore::MediaDeviceHashSalts&& salts) { m_deviceIdentifierHashSalts = WTFMove(salts); }
    const WebCore::MediaDeviceHashSalts& deviceIdentifierHashSalts() const { return m_deviceIdentifierHashSalts; }

    WebCore::CaptureDevice audioDevice() const { return m_eligibleAudioDevices.isEmpty() ? WebCore::CaptureDevice { } : m_eligibleAudioDevices[0]; }
    WebCore::CaptureDevice videoDevice() const { return m_eligibleVideoDevices.isEmpty() ? WebCore::CaptureDevice { } : m_eligibleVideoDevices[0]; }

#if ENABLE(MEDIA_STREAM)
    bool isUserGesturePriviledged() const { return m_request.isUserGesturePriviledged; }
#endif

    CompletionHandler<void(bool)> decisionCompletionHandler() { return std::exchange(m_decisionCompletionHandler, { }); }
    void setBeforeStartingCaptureCallback(Function<void()>&& callback) { m_beforeStartingCaptureCallback = WTFMove(callback); }
    Function<void()> beforeStartingCaptureCallback() { return std::exchange(m_beforeStartingCaptureCallback, { }); }

protected:
    UserMediaPermissionRequestProxy(UserMediaPermissionRequestManagerProxy&, std::optional<WebCore::UserMediaRequestIdentifier>, WebCore::FrameIdentifier mainFrameID, WebCore::FrameIdentifier, Ref<WebCore::SecurityOrigin>&& userMediaDocumentOrigin, Ref<WebCore::SecurityOrigin>&& topLevelDocumentOrigin, Vector<WebCore::CaptureDevice>&& audioDevices, Vector<WebCore::CaptureDevice>&& videoDevices, WebCore::MediaStreamRequest&&, CompletionHandler<void(bool)>&&);

    UserMediaPermissionRequestManagerProxy* manager() const;
    RefPtr<UserMediaPermissionRequestManagerProxy> protectedManager() const;

private:
    WeakPtr<UserMediaPermissionRequestManagerProxy> m_manager;
    Markable<WebCore::UserMediaRequestIdentifier> m_userMediaID;
    WebCore::FrameIdentifier m_mainFrameID;
    WebCore::FrameIdentifier m_frameID;
    Ref<WebCore::SecurityOrigin> m_userMediaDocumentSecurityOrigin;
    Ref<WebCore::SecurityOrigin> m_topLevelDocumentSecurityOrigin;
    Vector<WebCore::CaptureDevice> m_eligibleVideoDevices;
    Vector<WebCore::CaptureDevice> m_eligibleAudioDevices;
    WebCore::MediaStreamRequest m_request;
    bool m_hasPersistentAccess { false };
    WebCore::MediaDeviceHashSalts m_deviceIdentifierHashSalts;
    CompletionHandler<void(bool)> m_decisionCompletionHandler;
    Function<void()> m_beforeStartingCaptureCallback;
};

String convertEnumerationToString(UserMediaPermissionRequestProxy::UserMediaAccessDenialReason);

} // namespace WebKit

namespace WTF {

template<typename Type>
struct LogArgument;

template <>
struct LogArgument<WebKit::UserMediaPermissionRequestProxy::UserMediaAccessDenialReason> {
    static String toString(const WebKit::UserMediaPermissionRequestProxy::UserMediaAccessDenialReason type)
    {
        return convertEnumerationToString(type);
    }
};

}; // namespace WTF

