/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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
#include "WebPasteboardProxy.h"

#include "WebPasteboardProxyMessages.h"
#include "WebProcessProxy.h"
#include <WebCore/SharedMemory.h>
#include <mutex>
#include <wtf/CompletionHandler.h>
#include <wtf/NeverDestroyed.h>

#if !PLATFORM(COCOA)
#include <WebCore/PasteboardCustomData.h>
#include <WebCore/PasteboardItemInfo.h>
#endif

namespace WebKit {

#if PLATFORM(COCOA)
WebPasteboardProxy::PasteboardAccessInformation::~PasteboardAccessInformation() = default;
#endif

WebPasteboardProxy& WebPasteboardProxy::singleton()
{
    static std::once_flag onceFlag;
    static LazyNeverDestroyed<WebPasteboardProxy> proxy;

    std::call_once(onceFlag, [] {
        proxy.construct();
    });

    return proxy;
}

WebPasteboardProxy::WebPasteboardProxy()
{
}

void WebPasteboardProxy::addWebProcessProxy(WebProcessProxy& webProcessProxy)
{
    // FIXME: Can we handle all of these on a background queue?
    webProcessProxy.addMessageReceiver(Messages::WebPasteboardProxy::messageReceiverName(), *this);
    m_webProcessProxySet.add(webProcessProxy);
}
    
void WebPasteboardProxy::removeWebProcessProxy(WebProcessProxy& webProcessProxy)
{
    m_webProcessProxySet.remove(webProcessProxy);
}

RefPtr<WebProcessProxy> WebPasteboardProxy::webProcessProxyForConnection(IPC::Connection& connection) const
{
    for (Ref webProcessProxy : m_webProcessProxySet) {
        if (webProcessProxy->hasConnection(connection))
            return webProcessProxy.ptr();
    }
    return nullptr;
}

#if !PLATFORM(COCOA)

#if !PLATFORM(GTK)
void WebPasteboardProxy::typesSafeForDOMToReadAndWrite(IPC::Connection&, const String&, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(Vector<String>&&)>&& completionHandler)
{
    completionHandler({ });
}

void WebPasteboardProxy::writeCustomData(IPC::Connection&, const Vector<WebCore::PasteboardCustomData>&, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&& completionHandler)
{
    completionHandler(0);
}

void WebPasteboardProxy::allPasteboardItemInfo(IPC::Connection&, const String&, int64_t, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(std::optional<Vector<WebCore::PasteboardItemInfo>>&&)>&& completionHandler)
{
    completionHandler(std::nullopt);
}

void WebPasteboardProxy::informationForItemAtIndex(IPC::Connection&, size_t, const String&, int64_t, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(std::optional<WebCore::PasteboardItemInfo>&&)>&& completionHandler)
{
    completionHandler(std::nullopt);
}

void WebPasteboardProxy::getPasteboardItemsCount(IPC::Connection&, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(uint64_t)>&& completionHandler)
{
    completionHandler(0);
}

void WebPasteboardProxy::readURLFromPasteboard(IPC::Connection&, size_t, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(String&& url, String&& title)>&& completionHandler)
{
    completionHandler({ }, { });
}

void WebPasteboardProxy::readBufferFromPasteboard(IPC::Connection&, std::optional<size_t>, const String&, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(RefPtr<WebCore::SharedBuffer>&&)>&& completionHandler)
{
    completionHandler({ });
}
#endif

#if !USE(LIBWPE)

void WebPasteboardProxy::readStringFromPasteboard(IPC::Connection&, size_t, const String&, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(String&&)>&& completionHandler)
{
    completionHandler({ });
}

#endif // !USE(LIBWPE)

void WebPasteboardProxy::containsStringSafeForDOMToReadForType(IPC::Connection&, const String&, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(bool)>&& completionHandler)
{
    completionHandler(false);
}

void WebPasteboardProxy::containsURLStringSuitableForLoading(IPC::Connection&, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(bool)>&& completionHandler)
{
    completionHandler(false);
}

void WebPasteboardProxy::urlStringSuitableForLoading(IPC::Connection&, const String&, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(String&& url, String&& title)>&& completionHandler)
{
    completionHandler({ }, { });
}

#endif // !PLATFORM(COCOA)

} // namespace WebKit
