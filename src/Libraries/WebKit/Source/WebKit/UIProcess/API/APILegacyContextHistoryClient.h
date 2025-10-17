/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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
#ifndef APILegacyContextHistoryClient_h
#define APILegacyContextHistoryClient_h

#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
class WebFrameProxy;
class WebPageProxy;
class WebProcessPool;
struct WebNavigationDataStore;
}

namespace API {

class LegacyContextHistoryClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(LegacyContextHistoryClient);
public:
    virtual ~LegacyContextHistoryClient() { }

    virtual void didNavigateWithNavigationData(WebKit::WebProcessPool&, WebKit::WebPageProxy&, const WebKit::WebNavigationDataStore&, WebKit::WebFrameProxy&) { }
    virtual void didPerformClientRedirect(WebKit::WebProcessPool&, WebKit::WebPageProxy&, const WTF::String&, const WTF::String&, WebKit::WebFrameProxy&) { }
    virtual void didPerformServerRedirect(WebKit::WebProcessPool&, WebKit::WebPageProxy&, const WTF::String&, const WTF::String&, WebKit::WebFrameProxy&) { }
    virtual void didUpdateHistoryTitle(WebKit::WebProcessPool&, WebKit::WebPageProxy&, const WTF::String&, const WTF::String&, WebKit::WebFrameProxy&) { }
    virtual void populateVisitedLinks(WebKit::WebProcessPool&) { }
    virtual bool addsVisitedLinks() const { return false; }
};

} // namespace API

#endif // APILegacyContextHistoryClient_h
