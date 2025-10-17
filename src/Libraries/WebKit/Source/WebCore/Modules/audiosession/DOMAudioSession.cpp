/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
#include "DOMAudioSession.h"

#if ENABLE(DOM_AUDIO_SESSION)

#include "AudioSession.h"
#include "Document.h"
#include "EventNames.h"
#include "Page.h"
#include "PermissionsPolicy.h"
#include "PlatformMediaSessionManager.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DOMAudioSession);

static inline AudioSessionCategory fromDOMAudioSessionType(DOMAudioSession::Type type)
{
    switch (type) {
    case DOMAudioSession::Type::Auto:
        return AudioSessionCategory::None;
    case DOMAudioSession::Type::Playback:
        return AudioSessionCategory::MediaPlayback;
    case DOMAudioSession::Type::Transient:
        return AudioSessionCategory::AmbientSound;
    case DOMAudioSession::Type::TransientSolo:
        return AudioSessionCategory::SoloAmbientSound;
    case DOMAudioSession::Type::Ambient:
        return AudioSessionCategory::AmbientSound;
    case DOMAudioSession::Type::PlayAndRecord:
        return AudioSessionCategory::PlayAndRecord;
        break;
    };

    ASSERT_NOT_REACHED();
    return AudioSessionCategory::None;
}

Ref<DOMAudioSession> DOMAudioSession::create(ScriptExecutionContext* context)
{
    auto audioSession = adoptRef(*new DOMAudioSession(context));
    audioSession->suspendIfNeeded();
    return audioSession;
}

DOMAudioSession::DOMAudioSession(ScriptExecutionContext* context)
    : ActiveDOMObject(context)
{
    AudioSession::protectedSharedSession()->addInterruptionObserver(*this);
}

DOMAudioSession::~DOMAudioSession()
{
    AudioSession::protectedSharedSession()->removeInterruptionObserver(*this);
}

ExceptionOr<void> DOMAudioSession::setType(Type type)
{
    RefPtr document = downcast<Document>(scriptExecutionContext());
    if (!document)
        return Exception { ExceptionCode::InvalidStateError };

    RefPtr page = document->protectedPage();
    if (!page)
        return Exception { ExceptionCode::InvalidStateError };

    if (!PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Microphone, *document, PermissionsPolicy::ShouldReportViolation::No))
        return { };

    page->setAudioSessionType(type);

    auto categoryOverride = fromDOMAudioSessionType(type);
    AudioSession::protectedSharedSession()->setCategoryOverride(categoryOverride);

    if (categoryOverride == AudioSessionCategory::None)
        PlatformMediaSessionManager::updateAudioSessionCategoryIfNecessary();

    return { };
}

DOMAudioSession::Type DOMAudioSession::type() const
{
    RefPtr document = downcast<Document>(scriptExecutionContext());
    if (document && !PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Microphone, *document, PermissionsPolicy::ShouldReportViolation::No))
        return DOMAudioSession::Type::Auto;

    if (!document)
        return DOMAudioSession::Type::Auto;

    if (RefPtr page = document->protectedPage())
        return page->audioSessionType();

    return DOMAudioSession::Type::Auto;
}

static DOMAudioSession::State computeAudioSessionState()
{
    if (AudioSession::sharedSession().isInterrupted())
        return DOMAudioSession::State::Interrupted;

    if (!AudioSession::sharedSession().isActive())
        return DOMAudioSession::State::Inactive;

    return DOMAudioSession::State::Active;
}

DOMAudioSession::State DOMAudioSession::state() const
{
    RefPtr document = downcast<Document>(scriptExecutionContext());
    if (!document || !PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Microphone, *document, PermissionsPolicy::ShouldReportViolation::No))
        return DOMAudioSession::State::Inactive;

    if (!m_state)
        m_state = computeAudioSessionState();
    return *m_state;
}

void DOMAudioSession::stop()
{
}

bool DOMAudioSession::virtualHasPendingActivity() const
{
    return hasEventListeners(eventNames().statechangeEvent);
}

void DOMAudioSession::beginAudioSessionInterruption()
{
    scheduleStateChangeEvent();
}

void DOMAudioSession::endAudioSessionInterruption(AudioSession::MayResume)
{
    scheduleStateChangeEvent();
}

void DOMAudioSession::audioSessionActiveStateChanged()
{
    scheduleStateChangeEvent();
}

void DOMAudioSession::scheduleStateChangeEvent()
{
    RefPtr document = downcast<Document>(scriptExecutionContext());
    if (document && !PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Microphone, *document, PermissionsPolicy::ShouldReportViolation::No))
        return;

    if (m_hasScheduleStateChangeEvent)
        return;

    m_hasScheduleStateChangeEvent = true;
    queueTaskKeepingObjectAlive(*this, TaskSource::MediaElement, [this] {
        if (isContextStopped())
            return;

        m_hasScheduleStateChangeEvent = false;
        auto newState = computeAudioSessionState();

        if (m_state && *m_state == newState)
            return;

        m_state = newState;
        dispatchEvent(Event::create(eventNames().statechangeEvent, Event::CanBubble::No, Event::IsCancelable::No));
    });
}

} // namespace WebCore

#endif // ENABLE(DOM_AUDIO_SESSION)
