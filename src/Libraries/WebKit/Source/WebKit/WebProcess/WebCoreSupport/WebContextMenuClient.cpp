/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#include "WebContextMenuClient.h"

#if ENABLE(CONTEXT_MENUS)

#include "WebContextMenu.h"
#include "WebPage.h"
#include <WebCore/Editor.h>
#include <WebCore/FrameLoader.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/NotImplemented.h>
#include <WebCore/Page.h>
#include <WebCore/UserGestureIndicator.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebContextMenuClient);

void WebContextMenuClient::downloadURL(const URL&)
{
    // This is handled in the UI process.
}

#if !PLATFORM(COCOA)

void WebContextMenuClient::searchWithGoogle(const WebCore::LocalFrame* frame)
{
    auto page = frame->page();
    if (!page)
        return;

    auto searchString = frame->editor().selectedText().trim(deprecatedIsSpaceOrNewline);
    searchString = makeStringByReplacingAll(encodeWithURLEscapeSequences(searchString), "%20"_s, "+"_s);
    auto searchURL = URL { makeString("https://www.google.com/search?q="_s, searchString, "&ie=UTF-8&oe=UTF-8"_s) };

    WebCore::UserGestureIndicator indicator { WebCore::IsProcessingUserGesture::Yes };
    auto* localMainFrame = dynamicDowncast<WebCore::LocalFrame>(page->mainFrame());
    if (!localMainFrame)
        return;
    localMainFrame->loader().changeLocation(searchURL, { }, nullptr, WebCore::ReferrerPolicy::EmptyString, WebCore::ShouldOpenExternalURLsPolicy::ShouldNotAllow);
}

void WebContextMenuClient::lookUpInDictionary(WebCore::LocalFrame*)
{
    notImplemented();
}

bool WebContextMenuClient::isSpeaking() const
{
    notImplemented();
    return false;
}

void WebContextMenuClient::speak(const String&)
{
    notImplemented();
}

void WebContextMenuClient::stopSpeaking()
{
    notImplemented();
}

#endif

#if USE(ACCESSIBILITY_CONTEXT_MENUS)

void WebContextMenuClient::showContextMenu()
{
    m_page->contextMenu().show();
}

#endif

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
