/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
#include "UserMediaController.h"

#if ENABLE(MEDIA_STREAM)

#include "Document.h"
#include "LocalDOMWindow.h"
#include "PermissionsPolicy.h"
#include "RealtimeMediaSourceCenter.h"
#include "UserMediaRequest.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(UserMediaController);

ASCIILiteral UserMediaController::supplementName()
{
    return "UserMediaController"_s;
}

UserMediaController::UserMediaController(UserMediaClient* client)
    : m_client(client)
{
}

UserMediaController::~UserMediaController()
{
    m_client->pageDestroyed();
}

void provideUserMediaTo(Page* page, UserMediaClient* client)
{
    UserMediaController::provideTo(page, UserMediaController::supplementName(), makeUnique<UserMediaController>(client));
}

void UserMediaController::logGetUserMediaDenial(Document& document)
{
    if (RefPtr window = document.domWindow())
        window->printErrorMessage("Not allowed to call getUserMedia."_s);
}

void UserMediaController::logGetDisplayMediaDenial(Document& document)
{
    if (RefPtr window = document.domWindow())
        window->printErrorMessage("Not allowed to call getDisplayMedia."_s);
}

void UserMediaController::logEnumerateDevicesDenial(Document& document)
{
    // We redo the check to print to the console log.
    PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Camera, document);
    PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Microphone, document);
    if (RefPtr window = document.domWindow())
        window->printErrorMessage("Not allowed to call enumerateDevices."_s);
}

void UserMediaController::setShouldListenToVoiceActivity(Document& document, bool shouldListen)
{
    if (shouldListen) {
        ASSERT(!m_voiceActivityDocuments.contains(document));
        m_voiceActivityDocuments.add(document);
    } else {
        ASSERT(m_voiceActivityDocuments.contains(document));
        m_voiceActivityDocuments.remove(document);
    }
    checkDocumentForVoiceActivity(nullptr);
}

void UserMediaController::checkDocumentForVoiceActivity(const Document* document)
{
    if (document) {
        if (m_shouldListenToVoiceActivity == document->mediaState().containsAny(MediaProducer::IsCapturingAudioMask))
            return;
    }

    bool shouldListenToVoiceActivity = anyOf(m_voiceActivityDocuments, [] (auto& document) {
        return document.mediaState().containsAny(MediaProducer::IsCapturingAudioMask);
    });
    if (m_shouldListenToVoiceActivity == shouldListenToVoiceActivity)
        return;

    m_shouldListenToVoiceActivity = shouldListenToVoiceActivity;
    m_client->setShouldListenToVoiceActivity(m_shouldListenToVoiceActivity);
}

void UserMediaController::voiceActivityDetected()
{
#if ENABLE(MEDIA_SESSION)
    for (auto& document : m_voiceActivityDocuments)
        Ref(document)->voiceActivityDetected();
#endif
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
