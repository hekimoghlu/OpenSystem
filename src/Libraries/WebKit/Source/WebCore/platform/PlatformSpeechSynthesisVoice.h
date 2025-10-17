/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

class PlatformSpeechSynthesisVoice : public RefCounted<PlatformSpeechSynthesisVoice> {
public:
    WEBCORE_EXPORT static Ref<PlatformSpeechSynthesisVoice> create(const String& voiceURI, const String& name, const String& lang, bool localService, bool isDefault);
    static Ref<PlatformSpeechSynthesisVoice> create();

    const String& voiceURI() const { return m_voiceURI; }
    void setVoiceURI(const String& voiceURI) { m_voiceURI = voiceURI; }

    const String& name() const { return m_name; }
    void setName(const String& name) { m_name = name; }

    const String& lang() const { return m_lang; }
    void setLang(const String& lang) { m_lang = lang; }

    bool localService() const { return m_localService; }
    void setLocalService(bool localService) { m_localService = localService; }

    bool isDefault() const { return m_default; }
    void setIsDefault(bool isDefault) { m_default = isDefault; }

private:
    PlatformSpeechSynthesisVoice(const String& voiceURI, const String& name, const String& lang, bool localService, bool isDefault);
    PlatformSpeechSynthesisVoice() = default;

    String m_voiceURI;
    String m_name;
    String m_lang;
    bool m_localService { false };
    bool m_default { false };
};

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
