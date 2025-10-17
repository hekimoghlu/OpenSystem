/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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

#if ENABLE(CONTEXT_MENUS)

#include "ContextMenuContextData.h"
#include "UserData.h"
#include "WebContextMenuListenerProxy.h"
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS NSArray;
OBJC_CLASS NSMenu;

namespace WebKit {

class WebContextMenuItem;
class WebPageProxy;

class WebContextMenuProxy : public RefCounted<WebContextMenuProxy>, public WebContextMenuListenerProxyClient {
public:
    virtual ~WebContextMenuProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    virtual void show();

    WebPageProxy* page() const { return m_page.get(); }

#if PLATFORM(COCOA)
    virtual NSMenu *platformMenu() const = 0;
    virtual NSArray *platformData() const = 0;
#endif // PLATFORM(COCOA)

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
    virtual RetainPtr<CGImageRef> imageForCopySubject() const { return { }; }
#endif

protected:
    WebContextMenuProxy(WebPageProxy&, ContextMenuContextData&&, const UserData&);

    // WebContextMenuListenerProxyClient
    void useContextMenuItems(Vector<Ref<WebContextMenuItem>>&&) override;

    ContextMenuContextData m_context;
    const UserData m_userData;

private:
    virtual Vector<Ref<WebContextMenuItem>> proposedItems() const;
    virtual void showContextMenuWithItems(Vector<Ref<WebContextMenuItem>>&&) = 0;

    RefPtr<WebContextMenuListenerProxy> m_contextMenuListener;
    WeakPtr<WebPageProxy> m_page;
};

} // namespace WebKit

#endif
