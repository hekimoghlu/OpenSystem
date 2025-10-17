/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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

#include "SameDocumentNavigationType.h"
#include <WebCore/LayoutMilestone.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WallTime.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class DOMWindowExtension;
class DOMWrapperWorld;
class ResourceError;
class ResourceRequest;
class FragmentedSharedBuffer;
}

namespace WebKit {
class InjectedBundleBackForwardListItem;
class WebFrame;
class WebPage;
}

namespace API {
class Object;

namespace InjectedBundle {

class PageLoaderClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PageLoaderClient);
public:
    virtual ~PageLoaderClient() = default;

    virtual void willLoadURLRequest(WebKit::WebPage&, const WebCore::ResourceRequest&, API::Object*) { }
    virtual void willLoadDataRequest(WebKit::WebPage&, const WebCore::ResourceRequest&, RefPtr<WebCore::FragmentedSharedBuffer>, const WTF::String&, const WTF::String&, const WTF::URL&, API::Object*) { }

    virtual void didStartProvisionalLoadForFrame(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didReceiveServerRedirectForProvisionalLoadForFrame(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didFailProvisionalLoadWithErrorForFrame(WebKit::WebPage&, WebKit::WebFrame&, const WebCore::ResourceError&, RefPtr<API::Object>&) { }
    virtual void didCommitLoadForFrame(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didFinishDocumentLoadForFrame(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didFinishLoadForFrame(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didFinishProgress(WebKit::WebPage&) { }
    virtual void didFailLoadWithErrorForFrame(WebKit::WebPage&, WebKit::WebFrame&, const WebCore::ResourceError&, RefPtr<API::Object>&) { }
    virtual void didSameDocumentNavigationForFrame(WebKit::WebPage&, WebKit::WebFrame&, WebKit::SameDocumentNavigationType, RefPtr<API::Object>&) { }
    virtual void didReceiveTitleForFrame(WebKit::WebPage&, const WTF::String&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didRemoveFrameFromHierarchy(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didDisplayInsecureContentForFrame(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didRunInsecureContentForFrame(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }

    virtual void didFirstLayoutForFrame(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didFirstVisuallyNonEmptyLayoutForFrame(WebKit::WebPage&, WebKit::WebFrame&, RefPtr<API::Object>&) { }
    virtual void didLayoutForFrame(WebKit::WebPage&, WebKit::WebFrame&) { }
    virtual void didReachLayoutMilestone(WebKit::WebPage&, OptionSet<WebCore::LayoutMilestone>, RefPtr<API::Object>&) { }

    virtual void didClearWindowObjectForFrame(WebKit::WebPage&, WebKit::WebFrame&, WebCore::DOMWrapperWorld&) { }
    virtual void didCancelClientRedirectForFrame(WebKit::WebPage&, WebKit::WebFrame&) { }
    virtual void willPerformClientRedirectForFrame(WebKit::WebPage&, WebKit::WebFrame&, const WTF::String&, double /*delay*/, WallTime /*date*/) { }
    virtual void didHandleOnloadEventsForFrame(WebKit::WebPage&, WebKit::WebFrame&) { }

    virtual void globalObjectIsAvailableForFrame(WebKit::WebPage&, WebKit::WebFrame&, WebCore::DOMWrapperWorld&) { }
    virtual void serviceWorkerGlobalObjectIsAvailableForFrame(WebKit::WebPage&, WebKit::WebFrame&, WebCore::DOMWrapperWorld&) { }
    virtual void willDisconnectDOMWindowExtensionFromGlobalObject(WebKit::WebPage&, WebCore::DOMWindowExtension*) { }
    virtual void didReconnectDOMWindowExtensionToGlobalObject(WebKit::WebPage&, WebCore::DOMWindowExtension*) { }
    virtual void willDestroyGlobalObjectForDOMWindowExtension(WebKit::WebPage&, WebCore::DOMWindowExtension*) { }

    virtual void willInjectUserScriptForFrame(WebKit::WebPage&, WebKit::WebFrame&, WebCore::DOMWrapperWorld&) { }

    virtual bool shouldForceUniversalAccessFromLocalURL(WebKit::WebPage&, const WTF::String&) { return false; }

    virtual void featuresUsedInPage(WebKit::WebPage&, const Vector<WTF::String>&) { }

    virtual void willDestroyFrame(WebKit::WebPage&, WebKit::WebFrame&) { }

    virtual OptionSet<WebCore::LayoutMilestone> layoutMilestones() const { return { }; }
};

} // namespace InjectedBundle

} // namespace API
