/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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
#include "SpeechRecognition.h"

#include "ClientOrigin.h"
#include "Document.h"
#include "EventNames.h"
#include "FrameDestructionObserverInlines.h"
#include "Page.h"
#include "PermissionsPolicy.h"
#include "SpeechRecognitionError.h"
#include "SpeechRecognitionErrorEvent.h"
#include "SpeechRecognitionEvent.h"
#include "SpeechRecognitionResultData.h"
#include "SpeechRecognitionResultList.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SpeechRecognition);

Ref<SpeechRecognition> SpeechRecognition::create(Document& document)
{
    auto recognition = adoptRef(*new SpeechRecognition(document));
    recognition->suspendIfNeeded();
    return recognition;
}

SpeechRecognition::SpeechRecognition(Document& document)
    : ActiveDOMObject(document)
{
    if (auto* page = document.page()) {
        m_connection = &page->speechRecognitionConnection();
        m_connection->registerClient(*this);
    }
}

void SpeechRecognition::suspend(ReasonForSuspension)
{
    abortRecognition();
}

ExceptionOr<void> SpeechRecognition::startRecognition()
{
    if (m_state != State::Inactive)
        return Exception { ExceptionCode::InvalidStateError, "Recognition is being started or already started"_s };

    if (!m_connection)
        return Exception { ExceptionCode::UnknownError, "Recognition does not have a valid connection"_s };

    Ref document = downcast<Document>(*scriptExecutionContext());
    RefPtr frame = document->frame();
    if (!frame || !document->frameID())
        return Exception { ExceptionCode::UnknownError, "Recognition is not in a valid frame"_s };

    auto frameIdentifier = *document->frameID();
    if (!PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Microphone, document.get(), PermissionsPolicy::ShouldReportViolation::No)) {
        didError({ SpeechRecognitionErrorType::NotAllowed, "Permission is denied"_s });
        return { };
    }

    m_connection->start(identifier(), m_lang, m_continuous, m_interimResults, m_maxAlternatives, ClientOrigin { document->topOrigin().data(), document->securityOrigin().data() }, frameIdentifier);
    m_state = State::Starting;
    return { };
}

void SpeechRecognition::stopRecognition()
{
    if (m_state == State::Inactive || m_state == State::Stopping || m_state == State::Aborting)
        return;

    m_connection->stop(identifier());
    m_state = State::Stopping;
}

void SpeechRecognition::abortRecognition()
{
    if (m_state == State::Inactive || m_state == State::Aborting)
        return;

    m_connection->abort(identifier());
    m_state = State::Aborting;
}

void SpeechRecognition::stop()
{
    abortRecognition();

    if (!m_connection)
        return;
    m_connection->unregisterClient(*this);

    auto& document = downcast<Document>(*scriptExecutionContext());
    document.setActiveSpeechRecognition(nullptr);
}

void SpeechRecognition::didStart()
{
    if (m_state == State::Starting)
        m_state = State::Running;

    queueTaskToDispatchEvent(*this, TaskSource::Speech, Event::create(eventNames().startEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void SpeechRecognition::didStartCapturingAudio()
{
    auto& document = downcast<Document>(*scriptExecutionContext());
    document.setActiveSpeechRecognition(this);

    queueTaskToDispatchEvent(*this, TaskSource::Speech, Event::create(eventNames().audiostartEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void SpeechRecognition::didStartCapturingSound()
{
    queueTaskToDispatchEvent(*this, TaskSource::Speech, Event::create(eventNames().soundstartEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void SpeechRecognition::didStartCapturingSpeech()
{
    queueTaskToDispatchEvent(*this, TaskSource::Speech, Event::create(eventNames().speechstartEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void SpeechRecognition::didStopCapturingSpeech()
{
    queueTaskToDispatchEvent(*this, TaskSource::Speech, Event::create(eventNames().speechendEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void SpeechRecognition::didStopCapturingSound()
{
    queueTaskToDispatchEvent(*this, TaskSource::Speech, Event::create(eventNames().soundendEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void SpeechRecognition::didStopCapturingAudio()
{
    auto& document = downcast<Document>(*scriptExecutionContext());
    document.setActiveSpeechRecognition(nullptr);

    queueTaskToDispatchEvent(*this, TaskSource::Speech, Event::create(eventNames().audioendEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void SpeechRecognition::didFindNoMatch()
{
    queueTaskToDispatchEvent(*this, TaskSource::Speech, SpeechRecognitionEvent::create(eventNames().nomatchEvent, 0, nullptr));
}

void SpeechRecognition::didReceiveResult(Vector<SpeechRecognitionResultData>&& resultDatas)
{
    Vector<Ref<SpeechRecognitionResult>> allResults;
    allResults.reserveInitialCapacity(m_finalResults.size() + resultDatas.size());
    allResults.appendVector(m_finalResults);

    auto firstChangedIndex = allResults.size();
    for (auto resultData : resultDatas) {
        auto alternatives = WTF::map(resultData.alternatives, [](auto& alternativeData) {
            return SpeechRecognitionAlternative::create(WTFMove(alternativeData.transcript), alternativeData.confidence);
        });

        auto newResult = SpeechRecognitionResult::create(WTFMove(alternatives), resultData.isFinal);
        if (resultData.isFinal)
            m_finalResults.append(newResult);

        allResults.append(WTFMove(newResult));
    }

    auto resultList = SpeechRecognitionResultList::create(WTFMove(allResults));
    queueTaskToDispatchEvent(*this, TaskSource::Speech, SpeechRecognitionEvent::create(eventNames().resultEvent, firstChangedIndex, WTFMove(resultList)));
}

void SpeechRecognition::didError(const SpeechRecognitionError& error)
{
    m_finalResults.clear();
    m_state = State::Inactive;

    queueTaskToDispatchEvent(*this, TaskSource::Speech, SpeechRecognitionErrorEvent::create(eventNames().errorEvent, error.type, error.message));
}

void SpeechRecognition::didEnd()
{
    m_finalResults.clear();
    m_state = State::Inactive;

    queueTaskToDispatchEvent(*this, TaskSource::Speech, Event::create(eventNames().endEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

SpeechRecognition::~SpeechRecognition() = default;

bool SpeechRecognition::virtualHasPendingActivity() const
{
    return m_state != State::Inactive && hasEventListeners();
}

} // namespace WebCore
