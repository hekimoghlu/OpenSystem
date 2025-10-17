/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include "WebSpeechSynthesisClient.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include "WebSpeechSynthesisVoice.h"
#include <WebCore/Page.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(SPEECH_SYNTHESIS)

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSpeechSynthesisClient);

WebSpeechSynthesisClient::WebSpeechSynthesisClient(WebPage& page)
    : m_page(page)
{
}

const Vector<RefPtr<WebCore::PlatformSpeechSynthesisVoice>>& WebSpeechSynthesisClient::voiceList()
{
    RefPtr page = m_page.get();
    if (!page) {
        m_voices = { };
        return m_voices;
    }

    // FIXME: this message should not be sent synchronously. Instead, the UI process should
    // get the list of voices and pass it on to the WebContent processes, see
    // https://bugs.webkit.org/show_bug.cgi?id=195723
    auto sendResult = page->sendSync(Messages::WebPageProxy::SpeechSynthesisVoiceList());
    auto [voiceList] = sendResult.takeReplyOr(Vector<WebSpeechSynthesisVoice> { });

    m_voices = voiceList.map([](auto& voice) -> RefPtr<WebCore::PlatformSpeechSynthesisVoice> {
        return WebCore::PlatformSpeechSynthesisVoice::create(voice.voiceURI, voice.name, voice.lang, voice.localService, voice.defaultLang);
    });
    return m_voices;
}

WebCore::SpeechSynthesisClientObserver* WebSpeechSynthesisClient::corePageObserver() const
{
    RefPtr page = m_page.get();
    if (!page)
        return nullptr;

    RefPtr corePage = page->corePage();
    if (corePage && corePage->speechSynthesisClient() && corePage->speechSynthesisClient()->observer())
        return corePage->speechSynthesisClient()->observer().get();
    return nullptr;
}

void WebSpeechSynthesisClient::resetState()
{
    if (RefPtr page = m_page.get())
        page->send(Messages::WebPageProxy::SpeechSynthesisResetState());
}

void WebSpeechSynthesisClient::speak(RefPtr<WebCore::PlatformSpeechSynthesisUtterance> utterance)
{
    WTF::CompletionHandler<void()> startedCompletionHandler = [this, weakThis = WeakPtr { *this }]() mutable {
        if (!weakThis)
            return;
        if (auto observer = corePageObserver())
            observer->didStartSpeaking();
    };

    WTF::CompletionHandler<void()> finishedCompletionHandler = [this, weakThis = WeakPtr { *this }]() mutable {
        if (!weakThis)
            return;
        if (auto observer = corePageObserver())
            observer->didFinishSpeaking();
    };

    auto voice = utterance->voice();
    auto voiceURI = voice ? voice->voiceURI() : emptyString();
    auto name = voice ? voice->name() : emptyString();
    auto lang = voice ? voice->lang() : emptyString();
    auto localService = voice ? voice->localService() : false;
    auto isDefault = voice ? voice->isDefault() : false;

    RefPtr page = m_page.get();
    if (!page)
        return;

    page->sendWithAsyncReply(Messages::WebPageProxy::SpeechSynthesisSetFinishedCallback(), WTFMove(finishedCompletionHandler));
    page->sendWithAsyncReply(Messages::WebPageProxy::SpeechSynthesisSpeak(utterance->text(), utterance->lang(), utterance->volume(), utterance->rate(), utterance->pitch(), utterance->startTime(), voiceURI, name, lang, localService, isDefault), WTFMove(startedCompletionHandler));
}

void WebSpeechSynthesisClient::cancel()
{
    if (RefPtr page = m_page.get())
        page->send(Messages::WebPageProxy::SpeechSynthesisCancel());
}

void WebSpeechSynthesisClient::pause()
{
    RefPtr page = m_page.get();
    if (!page)
        return;

    WTF::CompletionHandler<void()> completionHandler = [this, weakThis = WeakPtr { *this }]() mutable {
        if (!weakThis)
            return;
        if (auto observer = corePageObserver())
            observer->didPauseSpeaking();
    };

    page->sendWithAsyncReply(Messages::WebPageProxy::SpeechSynthesisPause(), WTFMove(completionHandler));
}

void WebSpeechSynthesisClient::resume()
{
    RefPtr page = m_page.get();
    if (!page)
        return;

    WTF::CompletionHandler<void()> completionHandler = [this, weakThis = WeakPtr { *this }]() mutable {
        if (!weakThis)
            return;
        if (auto observer = corePageObserver())
            observer->didResumeSpeaking();
    };

    page->sendWithAsyncReply(Messages::WebPageProxy::SpeechSynthesisResume(), WTFMove(completionHandler));
}

} // namespace WebKit

#endif // ENABLE(SPEECH_SYNTHESIS)
