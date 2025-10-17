/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#ifndef WebContextMenuClient_h
#define WebContextMenuClient_h

#if ENABLE(CONTEXT_MENUS)

#include "WebPage.h"
#include <WebCore/ContextMenuClient.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebContextMenuClient : public WebCore::ContextMenuClient {
    WTF_MAKE_TZONE_ALLOCATED(WebContextMenuClient);
public:
    WebContextMenuClient(WebPage* page)
        : m_page(page)
    {
    }
    
private:
    void downloadURL(const URL&) override;
    void searchWithGoogle(const WebCore::LocalFrame*) override;
    void lookUpInDictionary(WebCore::LocalFrame*) override;
    bool isSpeaking() const override;
    void speak(const String&) override;
    void stopSpeaking() override;

#if ENABLE(IMAGE_ANALYSIS)
    bool supportsLookUpInImages() final { return true; }
#endif

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
    bool supportsCopySubject() final { return true; }
#endif

#if HAVE(TRANSLATION_UI_SERVICES)
    void handleTranslation(const WebCore::TranslationContextMenuInfo&) final;
#endif

#if PLATFORM(GTK)
    void insertEmoji(WebCore::LocalFrame&) override;
#endif

#if USE(ACCESSIBILITY_CONTEXT_MENUS)
    void showContextMenu() override;
#endif

    WeakPtr<WebPage> m_page;
};

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
#endif // WebContextMenuClient_h
