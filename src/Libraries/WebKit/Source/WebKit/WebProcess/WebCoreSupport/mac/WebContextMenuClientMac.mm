/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
#import "config.h"
#import "WebContextMenuClient.h"

#if ENABLE(CONTEXT_MENUS)

#import "MessageSenderInlines.h"
#import "WebPage.h"
#import "WebPageProxyMessages.h"
#import <WebCore/DictionaryLookup.h>
#import <WebCore/Editor.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/Page.h>
#import <WebCore/TextIndicator.h>
#import <WebCore/TranslationContextMenuInfo.h>
#import <wtf/text/WTFString.h>

namespace WebKit {
using namespace WebCore;

void WebContextMenuClient::lookUpInDictionary(LocalFrame* frame)
{
    m_page->performDictionaryLookupForSelection(*frame, frame->selection().selection(), TextIndicatorPresentationTransition::BounceAndCrossfade);
}

bool WebContextMenuClient::isSpeaking() const
{
    return m_page->isSpeaking();
}

void WebContextMenuClient::speak(const String&)
{
}

void WebContextMenuClient::stopSpeaking()
{
}

void WebContextMenuClient::searchWithGoogle(const LocalFrame* frame)
{
    auto searchString = frame->editor().selectedText().trim(deprecatedIsSpaceOrNewline);
    m_page->send(Messages::WebPageProxy::SearchTheWeb(searchString));
}

#if HAVE(TRANSLATION_UI_SERVICES)

void WebContextMenuClient::handleTranslation(const WebCore::TranslationContextMenuInfo& info)
{
    m_page->send(Messages::WebPageProxy::HandleContextMenuTranslation(info));
}

#endif // HAVE(TRANSLATION_UI_SERVICES)

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
