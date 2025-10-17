/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
#include "WebPageProxy.h"

#if ENABLE(VIDEO) && USE(GSTREAMER) && ENABLE(SPEECH_SYNTHESIS)

#include "MessageSenderInlines.h"
#include "WebPageMessages.h"
#include "WebPageProxyInternals.h"

namespace WebKit {

void WebPageProxy::Internals::didStartSpeaking(WebCore::PlatformSpeechSynthesisUtterance&)
{
    if (auto& handler = speechSynthesisData().speakingStartedCompletionHandler)
        handler();
}

void WebPageProxy::Internals::didFinishSpeaking(WebCore::PlatformSpeechSynthesisUtterance&)
{
    if (auto& handler = speechSynthesisData().speakingFinishedCompletionHandler)
        handler();
}

void WebPageProxy::Internals::didPauseSpeaking(WebCore::PlatformSpeechSynthesisUtterance&)
{
    if (auto& handler = speechSynthesisData().speakingPausedCompletionHandler)
        handler();
}

void WebPageProxy::Internals::didResumeSpeaking(WebCore::PlatformSpeechSynthesisUtterance&)
{
    if (auto& handler = speechSynthesisData().speakingResumedCompletionHandler)
        handler();
}

void WebPageProxy::Internals::speakingErrorOccurred(WebCore::PlatformSpeechSynthesisUtterance&)
{
    Ref protectedPage = page.get();
    protectedPage->protectedLegacyMainFrameProcess()->send(Messages::WebPage::SpeakingErrorOccurred(), protectedPage->webPageIDInMainFrameProcess());
}

void WebPageProxy::Internals::boundaryEventOccurred(WebCore::PlatformSpeechSynthesisUtterance&, WebCore::SpeechBoundary speechBoundary, unsigned charIndex, unsigned charLength)
{
    Ref protectedPage = page.get();
    protectedPage->protectedLegacyMainFrameProcess()->send(Messages::WebPage::BoundaryEventOccurred(speechBoundary == WebCore::SpeechBoundary::SpeechWordBoundary, charIndex, charLength), protectedPage->webPageIDInMainFrameProcess());
}

void WebPageProxy::Internals::voicesDidChange()
{
    Ref protectedPage = page.get();
    protectedPage->protectedLegacyMainFrameProcess()->send(Messages::WebPage::VoicesDidChange(), protectedPage->webPageIDInMainFrameProcess());
}

} // namespace WebKit

#endif // ENABLE(VIDEO) && USE(GSTREAMER) && ENABLE(SPEECH_SYNTHESIS)
