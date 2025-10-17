/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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

#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>

namespace WebCore {
class IntRect;
enum class TextDirection : bool;
}

namespace WebKit {

class NativeWebMouseEvent;

struct PlatformPopupMenuData;
struct WebPopupItem;

class WebPopupMenuProxy;

class WebPopupMenuProxyClient : public CanMakeCheckedPtr<WebPopupMenuProxyClient> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebPopupMenuProxyClient);
protected:
    virtual ~WebPopupMenuProxyClient() = default;

public:
    virtual void valueChangedForPopupMenu(WebPopupMenuProxy*, int32_t newSelectedIndex) = 0;
    virtual void setTextFromItemForPopupMenu(WebPopupMenuProxy*, int32_t index) = 0;
    virtual NativeWebMouseEvent* currentlyProcessedMouseDownEvent() = 0;
#if PLATFORM(GTK)
    virtual void failedToShowPopupMenu() = 0;
#endif
};

class WebPopupMenuProxy : public RefCounted<WebPopupMenuProxy> {
public:
    using Client = WebPopupMenuProxyClient;

    virtual ~WebPopupMenuProxy() = default;

    virtual void showPopupMenu(const WebCore::IntRect& rect, WebCore::TextDirection, double pageScaleFactor, const Vector<WebPopupItem>& items, const PlatformPopupMenuData&, int32_t selectedIndex) = 0;
    virtual void hidePopupMenu() = 0;
    virtual void cancelTracking() { }

    void invalidate() { m_client = nullptr; }

protected:
    explicit WebPopupMenuProxy(Client& client)
        : m_client(&client)
    {
    }

    Client* client() const { return m_client.get(); }

private:
    CheckedPtr<Client> m_client;
};

} // namespace WebKit
