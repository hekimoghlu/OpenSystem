/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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
#ifndef PlatformSpeechSynthesizerMock_h
#define PlatformSpeechSynthesizerMock_h

#if ENABLE(SPEECH_SYNTHESIS)

#include "PlatformSpeechSynthesizer.h"
#include "Timer.h"

namespace WebCore {

class PlatformSpeechSynthesizerMock : public PlatformSpeechSynthesizer {
public:
    WEBCORE_EXPORT static Ref<PlatformSpeechSynthesizer> create(PlatformSpeechSynthesizerClient&);

    virtual ~PlatformSpeechSynthesizerMock();
    virtual void speak(RefPtr<PlatformSpeechSynthesisUtterance>&&);
    virtual void pause();
    virtual void resume();
    virtual void cancel();

    void setUtteranceDuration(Seconds duration) { m_utteranceDuration = duration; }

private:
    explicit PlatformSpeechSynthesizerMock(PlatformSpeechSynthesizerClient&);

    virtual void initializeVoiceList();
    void speakingFinished();

    Timer m_speakingFinishedTimer;
    RefPtr<PlatformSpeechSynthesisUtterance> m_utterance;
    Seconds m_utteranceDuration { 100_ms };
};

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)

#endif // PlatformSpeechSynthesizer_h
