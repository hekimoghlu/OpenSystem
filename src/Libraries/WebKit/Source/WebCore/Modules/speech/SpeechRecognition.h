/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 6, 2023.
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

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "SpeechRecognitionConnection.h"
#include "SpeechRecognitionConnectionClient.h"
#include "SpeechRecognitionResult.h"

namespace WebCore {

class Document;
class SpeechRecognitionResult;

class SpeechRecognition final : public SpeechRecognitionConnectionClient, public ActiveDOMObject, public RefCounted<SpeechRecognition>, public EventTarget  {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SpeechRecognition);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<SpeechRecognition> create(Document&);

    USING_CAN_MAKE_WEAKPTR(SpeechRecognitionConnectionClient);

    const String& lang() const { return m_lang; }
    void setLang(String&& lang) { m_lang = WTFMove(lang); }

    bool continuous() const { return m_continuous; }
    void setContinuous(bool continuous) { m_continuous = continuous; }

    bool interimResults() const { return m_interimResults; }
    void setInterimResults(bool interimResults) { m_interimResults = interimResults; }

    uint64_t maxAlternatives() const { return m_maxAlternatives; }
    void setMaxAlternatives(unsigned maxAlternatives) { m_maxAlternatives = maxAlternatives; }

    ExceptionOr<void> startRecognition();
    void stopRecognition();
    void abortRecognition();

    virtual ~SpeechRecognition();

private:
    enum class State {
        Inactive,
        Starting,
        Running,
        Stopping,
        Aborting,
    };

    explicit SpeechRecognition(Document&);

    // SpeechRecognitionConnectionClient
    void didStart() final;
    void didStartCapturingAudio() final;
    void didStartCapturingSound() final;
    void didStartCapturingSpeech() final;
    void didStopCapturingSpeech() final;
    void didStopCapturingSound() final;
    void didStopCapturingAudio() final;
    void didFindNoMatch() final;
    void didReceiveResult(Vector<SpeechRecognitionResultData>&& resultDatas) final;
    void didError(const SpeechRecognitionError&) final;
    void didEnd() final;

    // ActiveDOMObject
    void suspend(ReasonForSuspension) final;
    void stop() final;

    // EventTarget
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::SpeechRecognition; }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    bool virtualHasPendingActivity() const final;

    String m_lang;
    bool m_continuous { false };
    bool m_interimResults { false };
    uint64_t m_maxAlternatives { 1 };

    State m_state { State::Inactive };
    Vector<Ref<SpeechRecognitionResult>> m_finalResults;
    RefPtr<SpeechRecognitionConnection> m_connection;
};

} // namespace WebCore
