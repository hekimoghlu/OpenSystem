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

#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>

OBJC_CLASS NSError;
OBJC_CLASS UIViewController;

namespace WebKit {
class WebPageProxy;
}

namespace API {

class FullscreenClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(FullscreenClient);
public:
    enum Type {
        APIType,
        WebKitType
    };

    virtual bool isType(Type target) const { return target == APIType; };

    virtual ~FullscreenClient() { }

    virtual void willEnterFullscreen(WebKit::WebPageProxy*) { }
    virtual void didEnterFullscreen(WebKit::WebPageProxy*) { }
    virtual void willExitFullscreen(WebKit::WebPageProxy*) { }
    virtual void didExitFullscreen(WebKit::WebPageProxy*) { }

#if PLATFORM(IOS_FAMILY)
    virtual void requestPresentingViewController(CompletionHandler<void(UIViewController *, NSError *)>&&) { }
#endif
};

} // namespace API
