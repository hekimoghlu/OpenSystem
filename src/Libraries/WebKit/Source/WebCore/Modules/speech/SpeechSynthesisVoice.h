/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class SpeechSynthesisVoice : public RefCounted<SpeechSynthesisVoice> {
public:
    virtual ~SpeechSynthesisVoice() = default;
    static Ref<SpeechSynthesisVoice> create(PlatformSpeechSynthesisVoice&);

    const String& voiceURI() const { return m_platformVoice->voiceURI(); }
    const String& name() const { return m_platformVoice->name(); }
    const String& lang() const { return m_platformVoice->lang(); }
    bool localService() const { return m_platformVoice->localService(); }
    bool isDefault() const { return m_platformVoice->isDefault(); }

    PlatformSpeechSynthesisVoice* platformVoice() { return m_platformVoice.ptr(); }

private:
    explicit SpeechSynthesisVoice(PlatformSpeechSynthesisVoice&);

    Ref<PlatformSpeechSynthesisVoice> m_platformVoice;
};

} // namespace WebCore

#endif // ENABLE(SPEECH_SYNTHESIS)
