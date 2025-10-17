/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#if ENABLE(SPEECH_SYNTHESIS)

#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>

namespace WebCore {

class PlatformSpeechSynthesisUtterance;
class SpeechSynthesisClientObserver;
class PlatformSpeechSynthesisVoice;
    
class SpeechSynthesisClient : public AbstractRefCountedAndCanMakeWeakPtr<SpeechSynthesisClient> {
public:
    virtual ~SpeechSynthesisClient() = default;

    virtual void setObserver(WeakPtr<SpeechSynthesisClientObserver>) = 0;
    virtual WeakPtr<SpeechSynthesisClientObserver> observer() const = 0;
    
    virtual const Vector<RefPtr<PlatformSpeechSynthesisVoice>>& voiceList() = 0;
    virtual void speak(RefPtr<PlatformSpeechSynthesisUtterance>) = 0;
    virtual void cancel() = 0;
    virtual void pause() = 0;
    virtual void resume() = 0;
    virtual void resetState() = 0;

};

class SpeechSynthesisClientObserver : public AbstractRefCountedAndCanMakeWeakPtr<SpeechSynthesisClientObserver>  {
public:
    virtual ~SpeechSynthesisClientObserver() = default;

    virtual void didStartSpeaking() = 0;
    virtual void didFinishSpeaking() = 0;
    virtual void didPauseSpeaking() = 0;
    virtual void didResumeSpeaking() = 0;
    virtual void speakingErrorOccurred() = 0;
    virtual void boundaryEventOccurred(bool wordBoundary, unsigned charIndex, unsigned charLength) = 0;
    virtual void voicesChanged() = 0;
};

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
