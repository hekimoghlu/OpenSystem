/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

#include <WebCore/PlatformSpeechSynthesisUtterance.h>
#include <WebCore/PlatformSpeechSynthesisVoice.h>
#include <WebCore/SpeechSynthesisClient.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebPage;
    
class WebSpeechSynthesisClient final : public RefCounted<WebSpeechSynthesisClient>, public WebCore::SpeechSynthesisClient {
    WTF_MAKE_TZONE_ALLOCATED(WebSpeechSynthesisClient);
public:
    static Ref<WebSpeechSynthesisClient> create(WebPage& webPage)
    {
        return adoptRef(*new WebSpeechSynthesisClient(webPage));
    }

    virtual ~WebSpeechSynthesisClient() { }

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    const Vector<RefPtr<WebCore::PlatformSpeechSynthesisVoice>>& voiceList() override;
    void speak(RefPtr<WebCore::PlatformSpeechSynthesisUtterance>) override;
    void cancel() override;
    void pause() override;
    void resume() override;

private:
    explicit WebSpeechSynthesisClient(WebPage&);

    void setObserver(WeakPtr<WebCore::SpeechSynthesisClientObserver> observer) override { m_observer = observer; }
    WeakPtr<WebCore::SpeechSynthesisClientObserver> observer() const override { return m_observer; }
    void resetState() override;

    WebCore::SpeechSynthesisClientObserver* corePageObserver() const;
    
    WeakPtr<WebPage> m_page;
    WeakPtr<WebCore::SpeechSynthesisClientObserver> m_observer;
    Vector<RefPtr<WebCore::PlatformSpeechSynthesisVoice>> m_voices;
};

} // namespace WebKit

#endif // ENABLE(SPEECH_SYNTHESIS)
