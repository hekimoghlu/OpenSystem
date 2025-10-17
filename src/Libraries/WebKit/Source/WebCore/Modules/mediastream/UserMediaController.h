/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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

#include "Page.h"
#include "UserMediaClient.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

class Exception;
class UserMediaRequest;

class UserMediaController : public Supplement<Page> {
    WTF_MAKE_TZONE_ALLOCATED(UserMediaController);
public:
    explicit UserMediaController(UserMediaClient*);
    ~UserMediaController();

    UserMediaClient* client() const { return m_client; }

    void requestUserMediaAccess(UserMediaRequest&);
    void cancelUserMediaAccessRequest(UserMediaRequest&);

    void enumerateMediaDevices(Document&, UserMediaClient::EnumerateDevicesCallback&&);

    UserMediaClient::DeviceChangeObserverToken addDeviceChangeObserver(Function<void()>&&);
    void removeDeviceChangeObserver(UserMediaClient::DeviceChangeObserverToken);

    void updateCaptureState(const Document&, bool isActive, MediaProducerMediaCaptureKind, CompletionHandler<void(std::optional<Exception>&&)>&&);

    void logGetUserMediaDenial(Document&);
    void logGetDisplayMediaDenial(Document&);
    void logEnumerateDevicesDenial(Document&);

    void setShouldListenToVoiceActivity(Document&, bool);
    void checkDocumentForVoiceActivity(const Document*);
    void voiceActivityDetected();

    WEBCORE_EXPORT static ASCIILiteral supplementName();
    static UserMediaController* from(Page* page) { return static_cast<UserMediaController*>(Supplement<Page>::from(page, supplementName())); }

private:
    UserMediaClient* m_client;

    WeakHashSet<Document, WeakPtrImplWithEventTargetData> m_voiceActivityDocuments;
    bool m_shouldListenToVoiceActivity { false };
};

inline void UserMediaController::requestUserMediaAccess(UserMediaRequest& request)
{
    m_client->requestUserMediaAccess(request);
}

inline void UserMediaController::cancelUserMediaAccessRequest(UserMediaRequest& request)
{
    m_client->cancelUserMediaAccessRequest(request);
}

inline void UserMediaController::enumerateMediaDevices(Document& document, UserMediaClient::EnumerateDevicesCallback&& completionHandler)
{
    m_client->enumerateMediaDevices(document, WTFMove(completionHandler));
}


inline UserMediaClient::DeviceChangeObserverToken UserMediaController::addDeviceChangeObserver(Function<void()>&& observer)
{
    return m_client->addDeviceChangeObserver(WTFMove(observer));
}

inline void UserMediaController::removeDeviceChangeObserver(UserMediaClient::DeviceChangeObserverToken token)
{
    m_client->removeDeviceChangeObserver(token);
}

inline void UserMediaController::updateCaptureState(const Document& document, bool isActive, MediaProducerMediaCaptureKind kind, CompletionHandler<void(std::optional<Exception>&&)>&& completionHandler)
{
    m_client->updateCaptureState(document, isActive, kind, WTFMove(completionHandler));
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
