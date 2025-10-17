/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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

#if PLATFORM(MAC)

#include "WebContextMenuProxy.h"
#include <wtf/RetainPtr.h>
#include <wtf/WeakObjCPtr.h>

OBJC_CLASS NSMenu;
OBJC_CLASS NSMenuItem;
OBJC_CLASS NSView;
OBJC_CLASS NSWindow;
OBJC_CLASS WKMenuDelegate;

#if ENABLE(WRITING_TOOLS)
namespace WebCore::WritingTools {
enum class RequestedTool : uint16_t;
}
#endif

namespace WebKit {

class WebContextMenuItemData;

class WebContextMenuProxyMac final : public WebContextMenuProxy {
public:
    static auto create(NSView *webView, WebPageProxy& page, ContextMenuContextData&& context, const UserData& userData)
    {
        return adoptRef(*new WebContextMenuProxyMac(webView, page, WTFMove(context), userData));
    }
    ~WebContextMenuProxyMac();

    void contextMenuItemSelected(const WebContextMenuItemData&);

#if ENABLE(WRITING_TOOLS)
    void handleContextMenuWritingTools(WebCore::WritingTools::RequestedTool);
#endif

    void handleShareMenuItem();

#if ENABLE(SERVICE_CONTROLS)
    void clearServicesMenu();
    void removeBackgroundFromControlledImage();
#endif

    NSWindow *window() const;

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
    RetainPtr<CGImageRef> imageForCopySubject() const final { return m_copySubjectResult; }
#endif

private:
    WebContextMenuProxyMac(NSView *, WebPageProxy&, ContextMenuContextData&&, const UserData&);

    void show() override;
    void showContextMenuWithItems(Vector<Ref<WebContextMenuItem>>&&) override;
    void useContextMenuItems(Vector<Ref<WebContextMenuItem>>&&) override;

    bool showAfterPostProcessingContextData();

    void getContextMenuItem(const WebContextMenuItemData&, CompletionHandler<void(NSMenuItem *)>&&);
    void getContextMenuFromItems(const Vector<WebContextMenuItemData>&, CompletionHandler<void(NSMenu *)>&&);

#if ENABLE(SERVICE_CONTROLS)
    enum class ShareMenuItemType : uint8_t { Placeholder, Popover };
    RetainPtr<NSMenuItem> createShareMenuItem(ShareMenuItemType);

    void showServicesMenu();
    void setupServicesMenu();
    void appendRemoveBackgroundItemToControlledImageMenuIfNeeded();
#endif

    NSMenu *platformMenu() const override;
    NSArray *platformData() const override;

    RetainPtr<NSMenu> m_menu;
    RetainPtr<WKMenuDelegate> m_menuDelegate;
    WeakObjCPtr<NSView> m_webView;
#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
    RetainPtr<CGImageRef> m_copySubjectResult;
#endif
};

} // namespace WebKit

#endif // PLATFORM(MAC)
