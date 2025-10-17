/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#include "WebMResourceClient.h"

#if ENABLE(ALTERNATE_WEBM_PLAYER)

#include "ResourceError.h"
#include "ResourceRequest.h"
#include "ResourceResponse.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebMResourceClient);

RefPtr<WebMResourceClient> WebMResourceClient::create(WebMResourceClientParent& parent, PlatformMediaResourceLoader& loader, ResourceRequest&& request)
{
    auto resource = loader.requestResource(WTFMove(request), PlatformMediaResourceLoader::LoadOption::DisallowCaching);
    if (!resource)
        return nullptr;
    auto client = adoptRef(*new WebMResourceClient { parent, Ref { *resource } });
    auto result = client.copyRef();
    resource->setClient(WTFMove(client));
    return result;
}

WebMResourceClient::WebMResourceClient(WebMResourceClientParent& parent, Ref<PlatformMediaResource>&& resource)
    : m_parent(parent)
    , m_resource(WTFMove(resource))
{
}

void WebMResourceClient::stop()
{
    if (!m_resource)
        return;

    auto resource = WTFMove(m_resource);
    resource->shutdown();
}

void WebMResourceClient::responseReceived(PlatformMediaResource&, const ResourceResponse& response, CompletionHandler<void(ShouldContinuePolicyCheck)>&& completionHandler)
{
    RefPtr parent = m_parent.get();
    if (parent)
        parent->dataLengthReceived(response.expectedContentLength());
    completionHandler(parent ? ShouldContinuePolicyCheck::Yes : ShouldContinuePolicyCheck::No);
}

void WebMResourceClient::dataReceived(PlatformMediaResource&, const SharedBuffer& buffer)
{
    if (RefPtr parent = m_parent.get())
        parent->dataReceived(buffer);
}

void WebMResourceClient::loadFailed(PlatformMediaResource&, const ResourceError& error)
{
    if (RefPtr parent = m_parent.get())
        parent->loadFailed(error);
}

void WebMResourceClient::loadFinished(PlatformMediaResource&, const NetworkLoadMetrics&)
{
    if (RefPtr parent = m_parent.get())
        parent->loadFinished();
}

} // namespace WebCore

#endif // ENABLE(ALTERNATE_WEBM_PLAYER)
