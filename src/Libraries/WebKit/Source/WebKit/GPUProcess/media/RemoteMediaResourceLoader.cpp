/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#include "RemoteMediaResourceLoader.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "RemoteMediaPlayerProxy.h"
#include <WebCore/ResourceError.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaResourceLoader);

using namespace WebCore;

RemoteMediaResourceLoader::RemoteMediaResourceLoader(RemoteMediaPlayerProxy& remoteMediaPlayerProxy)
    : m_remoteMediaPlayerProxy(remoteMediaPlayerProxy)
{
    ASSERT(isMainRunLoop());
}

RemoteMediaResourceLoader::~RemoteMediaResourceLoader()
{
}

RefPtr<PlatformMediaResource> RemoteMediaResourceLoader::requestResource(ResourceRequest&& request, LoadOptions options)
{
    ASSERT(isMainRunLoop());
    RefPtr remoteMediaPlayerProxy = m_remoteMediaPlayerProxy.get();
    if (!remoteMediaPlayerProxy)
        return nullptr;

    return remoteMediaPlayerProxy->requestResource(WTFMove(request), options);
}

void RemoteMediaResourceLoader::sendH2Ping(const URL& url, CompletionHandler<void(Expected<Seconds, ResourceError>&&)>&& completionHandler)
{
    ASSERT(isMainRunLoop());
    RefPtr remoteMediaPlayerProxy = m_remoteMediaPlayerProxy.get();
    if (!remoteMediaPlayerProxy)
        return completionHandler(makeUnexpected(internalError(url)));
    
    remoteMediaPlayerProxy->sendH2Ping(url, WTFMove(completionHandler));
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
