/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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
#ifndef InjectedBundlePageLoaderClient_h
#define InjectedBundlePageLoaderClient_h

#include "APIClient.h"
#include "APIInjectedBundlePageLoaderClient.h"
#include "WKBundlePageLoaderClient.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace API {
class Object;
class String;

template<> struct ClientTraits<WKBundlePageLoaderClientBase> {
    typedef std::tuple<WKBundlePageLoaderClientV0, WKBundlePageLoaderClientV1, WKBundlePageLoaderClientV2, WKBundlePageLoaderClientV3, WKBundlePageLoaderClientV4, WKBundlePageLoaderClientV5, WKBundlePageLoaderClientV6, WKBundlePageLoaderClientV7, WKBundlePageLoaderClientV8, WKBundlePageLoaderClientV9, WKBundlePageLoaderClientV10, WKBundlePageLoaderClientV11> Versions;
};
}

namespace WebKit {

class InjectedBundlePageLoaderClient : public API::Client<WKBundlePageLoaderClientBase>, public API::InjectedBundle::PageLoaderClient {
    WTF_MAKE_TZONE_ALLOCATED(InjectedBundlePageLoaderClient);
public:
    explicit InjectedBundlePageLoaderClient(const WKBundlePageLoaderClientBase*);

    void willLoadURLRequest(WebPage&, const WebCore::ResourceRequest&, API::Object*) override;
    void willLoadDataRequest(WebPage&, const WebCore::ResourceRequest&, RefPtr<WebCore::FragmentedSharedBuffer>, const WTF::String&, const WTF::String&, const URL&, API::Object*) override;

    void didStartProvisionalLoadForFrame(WebPage&, WebFrame&, RefPtr<API::Object>&) override;
    void didReceiveServerRedirectForProvisionalLoadForFrame(WebPage&, WebFrame&, RefPtr<API::Object>&) override;
    void didFailProvisionalLoadWithErrorForFrame(WebPage&, WebFrame&, const WebCore::ResourceError&, RefPtr<API::Object>&) override;
    void didCommitLoadForFrame(WebPage&, WebFrame&, RefPtr<API::Object>&) override;
    void didFinishDocumentLoadForFrame(WebPage&, WebFrame&, RefPtr<API::Object>&) override;
    void didFinishLoadForFrame(WebPage&, WebFrame&, RefPtr<API::Object>&) override;
    void didFinishProgress(WebPage&) override;
    void didFailLoadWithErrorForFrame(WebPage&, WebFrame&, const WebCore::ResourceError&, RefPtr<API::Object>&) override;
    void didSameDocumentNavigationForFrame(WebPage&, WebFrame&, SameDocumentNavigationType, RefPtr<API::Object>&) override;
    void didReceiveTitleForFrame(WebPage&, const WTF::String&, WebFrame&, RefPtr<API::Object>&) override;
    void didRemoveFrameFromHierarchy(WebPage&, WebFrame&, RefPtr<API::Object>&) override;
    void didDisplayInsecureContentForFrame(WebPage&, WebFrame&, RefPtr<API::Object>&) override;
    void didRunInsecureContentForFrame(WebPage&, WebFrame&, RefPtr<API::Object>&) override;

    void didFirstLayoutForFrame(WebPage&, WebFrame&, RefPtr<API::Object>&) override;
    void didFirstVisuallyNonEmptyLayoutForFrame(WebPage&, WebFrame&, RefPtr<API::Object>&) override;
    void didLayoutForFrame(WebPage&, WebFrame&) override;
    void didReachLayoutMilestone(WebPage&, OptionSet<WebCore::LayoutMilestone>, RefPtr<API::Object>&) override;

    void didClearWindowObjectForFrame(WebPage&, WebFrame&, WebCore::DOMWrapperWorld&) override;
    void didCancelClientRedirectForFrame(WebPage&, WebFrame&) override;
    void willPerformClientRedirectForFrame(WebPage&, WebFrame&, const WTF::String&, double /*delay*/, WallTime /*date*/) override;
    void didHandleOnloadEventsForFrame(WebPage&, WebFrame&) override;

    void globalObjectIsAvailableForFrame(WebPage&, WebFrame&, WebCore::DOMWrapperWorld&) override;
    void serviceWorkerGlobalObjectIsAvailableForFrame(WebPage&, WebFrame&, WebCore::DOMWrapperWorld&) override;
    void willDisconnectDOMWindowExtensionFromGlobalObject(WebPage&, WebCore::DOMWindowExtension*) override;
    void didReconnectDOMWindowExtensionToGlobalObject(WebPage&, WebCore::DOMWindowExtension*) override;
    void willDestroyGlobalObjectForDOMWindowExtension(WebPage&, WebCore::DOMWindowExtension*) override;

    void willInjectUserScriptForFrame(WebKit::WebPage&, WebKit::WebFrame&, WebCore::DOMWrapperWorld&) final;

    bool shouldForceUniversalAccessFromLocalURL(WebPage&, const WTF::String&) override;

    void featuresUsedInPage(WebPage&, const Vector<WTF::String>&) override;

    OptionSet<WebCore::LayoutMilestone> layoutMilestones() const override;
};

} // namespace WebKit

#endif // InjectedBundlePageLoaderClient_h
