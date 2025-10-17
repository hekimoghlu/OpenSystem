/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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

#include "APIData.h"
#include <WebCore/FrameLoaderTypes.h>
#include <WebCore/LayoutMilestone.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
class ResourceError;
}

namespace WebKit {
class AuthenticationChallengeProxy;
class WebBackForwardListItem;
class WebFrameProxy;
class WebPageProxy;
class WebProtectionSpace;
struct WebNavigationDataStore;
}

namespace API {

class Dictionary;
class Navigation;
class Object;

class LoaderClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(LoaderClient);
public:
    virtual ~LoaderClient() { }
    virtual void didStartProvisionalLoadForFrame(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, API::Navigation*, API::Object*) { }
    virtual void didReceiveServerRedirectForProvisionalLoadForFrame(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, API::Navigation*, API::Object*) { }
    virtual void didFailProvisionalLoadWithErrorForFrame(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, API::Navigation*, const WebCore::ResourceError&, API::Object*) { }
    virtual void didFinishLoadForFrame(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, API::Navigation*, API::Object*) { }
    virtual void didFailLoadWithErrorForFrame(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, API::Navigation*, const WebCore::ResourceError&, API::Object*) { }
    virtual void didFirstVisuallyNonEmptyLayoutForFrame(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, API::Object*) { }
    virtual void didReachLayoutMilestone(WebKit::WebPageProxy&, OptionSet<WebCore::LayoutMilestone>) { }
    virtual bool shouldKeepCurrentBackForwardListItemInList(WebKit::WebPageProxy&, WebKit::WebBackForwardListItem&) { return true; }
    virtual bool processDidCrash(WebKit::WebPageProxy&) { return false; }
    virtual void didChangeBackForwardList(WebKit::WebPageProxy&, WebKit::WebBackForwardListItem*, Vector<Ref<WebKit::WebBackForwardListItem>>&&) { }
    virtual void didCommitLoadForFrame(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, API::Navigation*, API::Object*) { }
};

} // namespace API
