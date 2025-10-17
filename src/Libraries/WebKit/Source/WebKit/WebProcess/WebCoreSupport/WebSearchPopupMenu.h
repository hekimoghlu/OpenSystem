/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#ifndef WebSearchPopupMenu_h
#define WebSearchPopupMenu_h

#include "WebPopupMenu.h"
#include <WebCore/SearchPopupMenu.h>

namespace WebKit {

class WebSearchPopupMenu : public WebCore::SearchPopupMenu {
public:
    static Ref<WebSearchPopupMenu> create(WebPage*, WebCore::PopupMenuClient*);

    WebCore::PopupMenu* popupMenu() override;
    void saveRecentSearches(const WTF::AtomString& name, const Vector<WebCore::RecentSearch>&) override;
    void loadRecentSearches(const WTF::AtomString& name, Vector<WebCore::RecentSearch>&) override;
    bool enabled() override;

private:
    WebSearchPopupMenu(WebPage*, WebCore::PopupMenuClient*);
    RefPtr<WebPopupMenu> protectedPopup();

    RefPtr<WebPopupMenu> m_popup;
};

}

#endif // WebSearchPopupMenu_h
