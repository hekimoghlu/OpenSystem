/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
#ifndef WebPopupMenu_h
#define WebPopupMenu_h

#include "WebPopupItem.h"
#include <WebCore/PopupMenu.h>
#include <wtf/Forward.h>
#include <wtf/Vector.h>

namespace WebCore {
class PopupMenuClient;
}

namespace WebKit {

class WebPage;
struct PlatformPopupMenuData;
struct WebPopupItem;

class WebPopupMenu : public WebCore::PopupMenu {
public:
    static Ref<WebPopupMenu> create(WebPage*, WebCore::PopupMenuClient*);
    ~WebPopupMenu();

    WebPage* page();

    void disconnectFromPage() { m_page = nullptr; }
    void didChangeSelectedIndex(int newIndex);
    void setTextForIndex(int newIndex);
#if PLATFORM(GTK)
    WebCore::PopupMenuClient* client() const { return m_popupClient.get(); }
#endif

    void show(const WebCore::IntRect&, WebCore::LocalFrameView&, int selectedIndex) override;
    void hide() override;
    void updateFromElement() override;
    void disconnectClient() override;

private:
    WebPopupMenu(WebPage*, WebCore::PopupMenuClient*);

    Vector<WebPopupItem> populateItems();
    void setUpPlatformData(const WebCore::IntRect& pageCoordinates, PlatformPopupMenuData&);

    CheckedPtr<WebCore::PopupMenuClient> m_popupClient;
    WeakPtr<WebPage> m_page;
};

} // namespace WebKit

#endif // WebPopupMenu_h
