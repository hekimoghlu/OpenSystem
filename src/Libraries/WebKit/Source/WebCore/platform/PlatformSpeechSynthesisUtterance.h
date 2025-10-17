/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#include <wtf/MonotonicTime.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class PlatformSpeechSynthesisUtteranceClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::PlatformSpeechSynthesisUtteranceClient> : std::true_type { };
}

namespace WebCore {
    
class PlatformSpeechSynthesisUtteranceClient : public CanMakeWeakPtr<PlatformSpeechSynthesisUtteranceClient> {
};
    
class PlatformSpeechSynthesisUtterance : public RefCounted<PlatformSpeechSynthesisUtterance> {
public:
    WEBCORE_EXPORT static Ref<PlatformSpeechSynthesisUtterance> create(PlatformSpeechSynthesisUtteranceClient&);

    const String& text() const { return m_text; }
    void setText(const String& text) { m_text = text; }
    
    const String& lang() const { return m_lang; }
    void setLang(const String& lang) { m_lang = lang; }
    
    PlatformSpeechSynthesisVoice* voice() const { return m_voice.get(); }
    void setVoice(PlatformSpeechSynthesisVoice* voice) { m_voice = voice; }

    // Range = [0, 1] where 1 is the default.
    float volume() const { return m_volume; }
    void setVolume(float volume) { m_volume = std::max(std::min(1.0f, volume), 0.0f); }
    
    // Range = [0.1, 10] where 1 is the default.
    float rate() const { return m_rate; }
    void setRate(float rate) { m_rate = std::max(std::min(10.0f, rate), 0.1f); }
    
    // Range = [0, 2] where 1 is the default.
    float pitch() const { return m_pitch; }
    void setPitch(float pitch) { m_pitch = std::max(std::min(2.0f, pitch), 0.0f); }

    MonotonicTime startTime() const { return m_startTime; }
    void setStartTime(MonotonicTime startTime) { m_startTime = startTime; }
    
    PlatformSpeechSynthesisUtteranceClient* client() const { return m_client.get(); }
    void setClient(PlatformSpeechSynthesisUtteranceClient* client) { m_client = client; }

#if PLATFORM(COCOA)
    id wrapper() const { return m_wrapper.get(); }
    void setWrapper(id utterance) { m_wrapper = utterance; }
#endif

private:
    explicit PlatformSpeechSynthesisUtterance(PlatformSpeechSynthesisUtteranceClient&);

    WeakPtr<PlatformSpeechSynthesisUtteranceClient> m_client;
    String m_text;
    String m_lang;
    RefPtr<PlatformSpeechSynthesisVoice> m_voice;
    float m_volume { 1 };
    float m_rate { 1 };
    float m_pitch { 1 };
    MonotonicTime m_startTime;

#if PLATFORM(COCOA)
    RetainPtr<id> m_wrapper;
#endif
};
    
} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
