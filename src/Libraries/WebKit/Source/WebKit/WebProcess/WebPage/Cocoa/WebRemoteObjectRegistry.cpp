/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
#include "WebRemoteObjectRegistry.h"

#include "RemoteObjectRegistryMessages.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebRemoteObjectRegistry);

WebRemoteObjectRegistry::WebRemoteObjectRegistry(_WKRemoteObjectRegistry *remoteObjectRegistry, WebPage& page)
    : RemoteObjectRegistry(remoteObjectRegistry)
    , m_page(page)
{
    WebProcess::singleton().addMessageReceiver(Messages::RemoteObjectRegistry::messageReceiverName(), page.identifier(), *this);
    page.setRemoteObjectRegistry(this);
}

WebRemoteObjectRegistry::~WebRemoteObjectRegistry()
{
    close();
}

void WebRemoteObjectRegistry::close()
{
    RefPtr page = m_page.get();
    if (!page)
        return;

    if (page->remoteObjectRegistry() == this) {
        WebProcess::singleton().removeMessageReceiver(Messages::RemoteObjectRegistry::messageReceiverName(), page->identifier());
        page->setRemoteObjectRegistry(nullptr);
    }
}

auto WebRemoteObjectRegistry::messageSender() -> std::optional<MessageSender>
{
    if (m_page)
        return *m_page;
    return std::nullopt;
}

std::optional<uint64_t> WebRemoteObjectRegistry::messageDestinationID()
{
    if (m_page)
        return m_page->webPageProxyIdentifier().toUInt64();
    return std::nullopt;
}

} // namespace WebKit
