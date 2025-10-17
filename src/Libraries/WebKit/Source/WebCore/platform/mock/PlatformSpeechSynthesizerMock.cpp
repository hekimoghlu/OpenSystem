/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#include "PlatformSpeechSynthesizerMock.h"
#include "PlatformSpeechSynthesisUtterance.h"

#if ENABLE(SPEECH_SYNTHESIS)

namespace WebCore {

Ref<PlatformSpeechSynthesizer> PlatformSpeechSynthesizerMock::create(PlatformSpeechSynthesizerClient& client)
{
    return adoptRef(*new PlatformSpeechSynthesizerMock(client));
}

PlatformSpeechSynthesizerMock::PlatformSpeechSynthesizerMock(PlatformSpeechSynthesizerClient& client)
    : PlatformSpeechSynthesizer(client)
    , m_speakingFinishedTimer(*this, &PlatformSpeechSynthesizerMock::speakingFinished)
{
}

PlatformSpeechSynthesizerMock::~PlatformSpeechSynthesizerMock() = default;

void PlatformSpeechSynthesizerMock::speakingFinished()
{
    ASSERT(m_utterance.get());
    RefPtr<PlatformSpeechSynthesisUtterance> protect(m_utterance);
    m_utterance = nullptr;

    client().didFinishSpeaking(*protect);
}

void PlatformSpeechSynthesizerMock::initializeVoiceList()
{
    m_voiceList.append(PlatformSpeechSynthesisVoice::create("mock.voice.bruce"_s, "bruce"_s, "en-US"_s, true, true));
    m_voiceList.append(PlatformSpeechSynthesisVoice::create("mock.voice.clark"_s, "clark"_s, "en-US"_s, true, false));
    m_voiceList.append(PlatformSpeechSynthesisVoice::create("mock.voice.logan"_s, "logan"_s, "fr-CA"_s, true, true));
}

void PlatformSpeechSynthesizerMock::speak(RefPtr<PlatformSpeechSynthesisUtterance>&& utterance)
{
    ASSERT(!m_utterance);
    m_utterance = WTFMove(utterance);
    client().didStartSpeaking(*m_utterance);

    // Fire a fake word and then sentence boundary event. Since the entire sentence is the full length, pick arbitrary (3) length for the word.
    client().boundaryEventOccurred(*m_utterance, SpeechBoundary::SpeechWordBoundary, 0, 3);
    client().boundaryEventOccurred(*m_utterance, SpeechBoundary::SpeechSentenceBoundary, 0, m_utterance->text().length());

    // Give the fake speech job some time so that pause and other functions have time to be called.
    m_speakingFinishedTimer.startOneShot(m_utteranceDuration);
}

void PlatformSpeechSynthesizerMock::cancel()
{
    if (!m_utterance)
        return;

    m_speakingFinishedTimer.stop();
    auto utterance = std::exchange(m_utterance, nullptr);
    client().speakingErrorOccurred(*utterance);
}

void PlatformSpeechSynthesizerMock::pause()
{
    if (!m_utterance)
        return;

    client().didPauseSpeaking(*m_utterance);
}

void PlatformSpeechSynthesizerMock::resume()
{
    if (!m_utterance)
        return;

    client().didResumeSpeaking(*m_utterance);
}

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
