/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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

#include "APIObject.h"
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class WebContextMenuItem;

struct WebContextMenuListenerProxyClient : AbstractRefCountedAndCanMakeWeakPtr<WebContextMenuListenerProxyClient> {
    virtual ~WebContextMenuListenerProxyClient() = default;

    virtual void useContextMenuItems(Vector<Ref<WebContextMenuItem>>&&) = 0;
};

class WebContextMenuListenerProxy : public API::ObjectImpl<API::Object::Type::ContextMenuListener> {
public:
    static Ref<WebContextMenuListenerProxy> create(WebContextMenuListenerProxyClient& client)
    {
        return adoptRef(*new WebContextMenuListenerProxy(client));
    }

    virtual ~WebContextMenuListenerProxy();

    void useContextMenuItems(Vector<Ref<WebContextMenuItem>>&&);

private:
    explicit WebContextMenuListenerProxy(WebContextMenuListenerProxyClient&);

    WeakPtr<WebContextMenuListenerProxyClient> m_client;
};

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
