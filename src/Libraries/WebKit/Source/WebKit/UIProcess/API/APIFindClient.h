/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#ifndef APIFindClient_h
#define APIFindClient_h

#include <WebCore/PlatformLayer.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS CALayer;

namespace WebCore {
class IntRect;
}

namespace WebKit {
class WebPageProxy;
}

namespace API {

class FindClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(FindClient);
public:
    virtual ~FindClient() { }

    virtual void didCountStringMatches(WebKit::WebPageProxy*, const WTF::String&, uint32_t) { }
    virtual void didFindString(WebKit::WebPageProxy*, const WTF::String&, const Vector<WebCore::IntRect>& matchRects, uint32_t, int32_t, bool didWrapAround) { }
    virtual void didFailToFindString(WebKit::WebPageProxy*, const WTF::String&) { }

    virtual void didAddLayerForFindOverlay(WebKit::WebPageProxy*, PlatformLayer*) { }
    virtual void didRemoveLayerForFindOverlay(WebKit::WebPageProxy*) { }
};

} // namespace API

#endif // APIFindClient_h
