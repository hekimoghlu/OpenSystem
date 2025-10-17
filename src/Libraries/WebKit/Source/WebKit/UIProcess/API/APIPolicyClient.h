/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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

#include "WebFramePolicyListenerProxy.h"
#include <WebCore/FrameLoaderTypes.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
}

namespace WebKit {
struct NavigationActionData;
class WebPageProxy;
class WebFrameProxy;
class WebFramePolicyListenerProxy;
}

namespace API {
class Object;

class PolicyClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PolicyClient);
public:
    virtual ~PolicyClient() { }

    virtual void decidePolicyForNavigationAction(WebKit::WebPageProxy&, WebKit::WebFrameProxy*, Ref<API::NavigationAction>&&, WebKit::WebFrameProxy*, const WebCore::ResourceRequest&, const WebCore::ResourceRequest&, Ref<WebKit::WebFramePolicyListenerProxy>&& listener)
    {
        listener->use();
    }
    virtual void decidePolicyForNewWindowAction(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, Ref<API::NavigationAction>&&, const WebCore::ResourceRequest&, const WTF::String&, Ref<WebKit::WebFramePolicyListenerProxy>&& listener)
    {
        listener->use();
    }
    virtual void decidePolicyForResponse(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, const WebCore::ResourceResponse&, const WebCore::ResourceRequest&, bool, Ref<WebKit::WebFramePolicyListenerProxy>&& listener)
    {
        listener->use();
    }
};

} // namespace API
