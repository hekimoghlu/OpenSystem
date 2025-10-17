/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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

#include "PlatformSpeechSynthesisVoice.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

#if PLATFORM(COCOA)
#include <wtf/RetainPtr.h>
OBJC_CLASS WebSpeechSynthesisWrapper;
#endif

namespace WebCore {

#if USE(SPIEL)
class SpielSpeechWrapper;
#endif

enum class SpeechBoundary : uint8_t {
    SpeechWordBoundary,
    SpeechSentenceBoundary
};

#if USE(FLITE) && USE(GSTREAMER)
class GstSpeechSynthesisWrapper;
#endif
class PlatformSpeechSynthesisUtterance;

class PlatformSpeechSynthesizerClient {
public:
    virtual void didStartSpeaking(PlatformSpeechSynthesisUtterance&) = 0;
    virtual void didFinishSpeaking(PlatformSpeechSynthesisUtterance&) = 0;
    virtual void didPauseSpeaking(PlatformSpeechSynthesisUtterance&) = 0;
    virtual void didResumeSpeaking(PlatformSpeechSynthesisUtterance&) = 0;
    virtual void speakingErrorOccurred(PlatformSpeechSynthesisUtterance&) = 0;
    virtual void boundaryEventOccurred(PlatformSpeechSynthesisUtterance&, SpeechBoundary, unsigned charIndex, unsigned charLength) = 0;
    virtual void voicesDidChange() = 0;
protected:
    virtual ~PlatformSpeechSynthesizerClient() = default;
};

class WEBCORE_EXPORT PlatformSpeechSynthesizer : public RefCounted<PlatformSpeechSynthesizer> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PlatformSpeechSynthesizer, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static Ref<PlatformSpeechSynthesizer> create(PlatformSpeechSynthesizerClient&);

    // FIXME: We have multiple virtual functions just so we can support a mock for testing.
    // Seems wasteful. Would be nice to find a better way.
    WEBCORE_EXPORT virtual ~PlatformSpeechSynthesizer();

    const Vector<RefPtr<PlatformSpeechSynthesisVoice>>& voiceList() const;
    virtual void speak(RefPtr<PlatformSpeechSynthesisUtterance>&&);
    virtual void pause();
    virtual void resume();
    virtual void cancel();
    virtual void resetState();
    virtual void voicesDidChange();

    PlatformSpeechSynthesizerClient& client() const { return m_speechSynthesizerClient; }

protected:
    explicit PlatformSpeechSynthesizer(PlatformSpeechSynthesizerClient&);
    Vector<RefPtr<PlatformSpeechSynthesisVoice>> m_voiceList;

private:
    virtual void initializeVoiceList();
    virtual void resetVoiceList();

    bool m_voiceListIsInitialized { false };
    PlatformSpeechSynthesizerClient& m_speechSynthesizerClient;

#if PLATFORM(COCOA)
    RetainPtr<WebSpeechSynthesisWrapper> m_platformSpeechWrapper;
#elif USE(FLITE) && USE(GSTREAMER)
    std::unique_ptr<GstSpeechSynthesisWrapper> m_platformSpeechWrapper;
#elif USE(SPIEL)
    std::unique_ptr<SpielSpeechWrapper> m_platformSpeechWrapper;
#endif
};

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
