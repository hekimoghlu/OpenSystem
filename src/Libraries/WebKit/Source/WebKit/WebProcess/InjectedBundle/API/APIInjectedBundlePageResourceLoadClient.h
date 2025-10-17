/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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

#include <WebCore/ResourceLoaderIdentifier.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
class WebFrame;
class WebPage;
}

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
}

namespace API {

namespace InjectedBundle {

class ResourceLoadClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ResourceLoadClient);
public:
    virtual ~ResourceLoadClient() = default;

    virtual void didInitiateLoadForResource(WebKit::WebPage&, WebKit::WebFrame&, WebCore::ResourceLoaderIdentifier, const WebCore::ResourceRequest&, bool /*pageIsProvisionallyLoading*/) { }
    virtual void willSendRequestForFrame(WebKit::WebPage&, WebKit::WebFrame&, WebCore::ResourceLoaderIdentifier, WebCore::ResourceRequest&, const WebCore::ResourceResponse&) { }
    virtual void didReceiveResponseForResource(WebKit::WebPage&, WebKit::WebFrame&, WebCore::ResourceLoaderIdentifier, const WebCore::ResourceResponse&) { }
    virtual void didReceiveContentLengthForResource(WebKit::WebPage&, WebKit::WebFrame&, WebCore::ResourceLoaderIdentifier, uint64_t contentLength) { }
    virtual void didFinishLoadForResource(WebKit::WebPage&, WebKit::WebFrame&, WebCore::ResourceLoaderIdentifier) { }
    virtual void didFailLoadForResource(WebKit::WebPage&, WebKit::WebFrame&, WebCore::ResourceLoaderIdentifier, const WebCore::ResourceError&) { }
    virtual bool shouldCacheResponse(WebKit::WebPage&, WebKit::WebFrame&, WebCore::ResourceLoaderIdentifier) { return true; }
    virtual bool shouldUseCredentialStorage(WebKit::WebPage&, WebKit::WebFrame&, WebCore::ResourceLoaderIdentifier) { return true; }
};

} // namespace InjectedBundle

} // namespace API
