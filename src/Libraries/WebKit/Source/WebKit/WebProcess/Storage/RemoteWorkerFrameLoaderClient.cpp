/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include "RemoteWorkerFrameLoaderClient.h"

#include "Logging.h"
#include <WebCore/DocumentLoader.h>

namespace WebKit {

RemoteWorkerFrameLoaderClient::RemoteWorkerFrameLoaderClient(WebCore::FrameLoader& frameLoader, WebPageProxyIdentifier webPageProxyID, WebCore::PageIdentifier pageID, const String& userAgent)
    : WebCore::EmptyFrameLoaderClient(frameLoader)
    , m_webPageProxyID(webPageProxyID)
    , m_userAgent(userAgent)
{
    RELEASE_LOG(Worker, "RemoteWorkerFrameLoaderClient::RemoteWorkerFrameLoaderClient webPageProxyID %" PRIu64 ", pageID %" PRIu64, webPageProxyID.toUInt64(), pageID.toUInt64());
}

Ref<WebCore::DocumentLoader> RemoteWorkerFrameLoaderClient::createDocumentLoader(const WebCore::ResourceRequest& request, const WebCore::SubstituteData& substituteData)
{
    return WebCore::DocumentLoader::create(request, substituteData);
}

} // namespace WebKit
