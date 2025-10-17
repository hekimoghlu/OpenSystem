/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#ifndef InjectedBundlePageResourceLoadClient_h
#define InjectedBundlePageResourceLoadClient_h

#include "APIClient.h"
#include "APIInjectedBundlePageResourceLoadClient.h"
#include "WKBundlePageResourceLoadClient.h"
#include <wtf/TZoneMalloc.h>

namespace API {
template<> struct ClientTraits<WKBundlePageResourceLoadClientBase> {
    typedef std::tuple<WKBundlePageResourceLoadClientV0, WKBundlePageResourceLoadClientV1> Versions;
};
}

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
}

namespace WebKit {

class WebPage;
class WebFrame;

class InjectedBundlePageResourceLoadClient : public API::InjectedBundle::ResourceLoadClient, public API::Client<WKBundlePageResourceLoadClientBase> {
    WTF_MAKE_TZONE_ALLOCATED(InjectedBundlePageResourceLoadClient);
public:
    explicit InjectedBundlePageResourceLoadClient(const WKBundlePageResourceLoadClientBase*);

    void didInitiateLoadForResource(WebPage&, WebFrame&, WebCore::ResourceLoaderIdentifier, const WebCore::ResourceRequest&, bool /*pageIsProvisionallyLoading*/) override;
    void willSendRequestForFrame(WebPage&, WebFrame&, WebCore::ResourceLoaderIdentifier, WebCore::ResourceRequest&, const WebCore::ResourceResponse&) override;
    void didReceiveResponseForResource(WebPage&, WebFrame&, WebCore::ResourceLoaderIdentifier, const WebCore::ResourceResponse&) override;
    void didReceiveContentLengthForResource(WebPage&, WebFrame&, WebCore::ResourceLoaderIdentifier, uint64_t contentLength) override;
    void didFinishLoadForResource(WebPage&, WebFrame&, WebCore::ResourceLoaderIdentifier) override;
    void didFailLoadForResource(WebPage&, WebFrame&, WebCore::ResourceLoaderIdentifier, const WebCore::ResourceError&) override;
    bool shouldCacheResponse(WebPage&, WebFrame&, WebCore::ResourceLoaderIdentifier) override;
    bool shouldUseCredentialStorage(WebPage&, WebFrame&, WebCore::ResourceLoaderIdentifier) override;
};

} // namespace WebKit

#endif // InjectedBundlePageResourceLoadClient_h
