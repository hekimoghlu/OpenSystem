/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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

#include "WebPopupMenuProxy.h"
#include <WebCore/GUniquePtrGtk.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/WTFString.h>

typedef struct _GdkDevice GdkDevice;
typedef struct _GtkTreePath GtkTreePath;
typedef struct _GtkTreeView GtkTreeView;
typedef struct _GtkTreeViewColumn GtkTreeViewColumn;
#if USE(GTK4)
typedef struct _GdkEvent GdkEvent;
#else
typedef union _GdkEvent GdkEvent;
#endif

namespace WebCore {
class IntRect;
}

namespace WebKit {

class WebPageProxy;

class WebPopupMenuProxyGtk : public WebPopupMenuProxy {
public:
    static Ref<WebPopupMenuProxyGtk> create(GtkWidget* webView, WebPopupMenuProxy::Client& client)
    {
        return adoptRef(*new WebPopupMenuProxyGtk(webView, client));
    }
    ~WebPopupMenuProxyGtk();

    void showPopupMenu(const WebCore::IntRect&, WebCore::TextDirection, double pageScaleFactor, const Vector<WebPopupItem>&, const PlatformPopupMenuData&, int32_t selectedIndex) override;
    void hidePopupMenu() override;
    void cancelTracking() override;

    virtual void selectItem(unsigned itemIndex);
    virtual void activateItem(std::optional<unsigned> itemIndex);

    bool handleKeyPress(unsigned keyval, uint32_t timestamp);
    void activateSelectedItem();

protected:
    WebPopupMenuProxyGtk(GtkWidget*, WebPopupMenuProxy::Client&);

    GtkWidget* m_webView { nullptr };

private:
    void createPopupMenu(const Vector<WebPopupItem>&, int32_t selectedIndex);
    void show();
    bool activateItemAtPath(GtkTreePath*);
    std::optional<unsigned> typeAheadFindIndex(unsigned keyval, uint32_t timestamp);
    bool typeAheadFind(unsigned keyval, uint32_t timestamp);

#if !USE(GTK4)
    static gboolean buttonPressEventCallback(GtkWidget*, GdkEventButton*, WebPopupMenuProxyGtk*);
    static gboolean keyPressEventCallback(GtkWidget*, GdkEvent*, WebPopupMenuProxyGtk*);
    static gboolean treeViewButtonReleaseEventCallback(GtkWidget*, GdkEvent*, WebPopupMenuProxyGtk*);
#endif

    GtkWidget* m_popup { nullptr };
    GtkWidget* m_treeView { nullptr };
#if !USE(GTK4)
    GdkDevice* m_device { nullptr };
#endif

    Vector<GUniquePtr<GtkTreePath>> m_paths;
    std::optional<unsigned> m_selectedItem;

    // Typeahead find.
    gunichar m_repeatingCharacter { '\0' };
    uint32_t m_previousKeyEventTime { 0 };
    GString* m_currentSearchString { nullptr };
};

} // namespace WebKit
