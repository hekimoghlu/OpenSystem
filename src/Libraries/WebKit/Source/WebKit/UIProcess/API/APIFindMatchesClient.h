/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#ifndef APIFindMatchesClient_h
#define APIFindMatchesClient_h

#include <WebCore/IntRect.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
class WebImage;
class WebPageProxy;
}

namespace API {

class FindMatchesClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(FindMatchesClient);
public:
    virtual ~FindMatchesClient() { }

    virtual void didFindStringMatches(WebKit::WebPageProxy*, const WTF::String&, const WTF::Vector<WTF::Vector<WebCore::IntRect>>&, int32_t) { }
    virtual void didGetImageForMatchResult(WebKit::WebPageProxy*, WebKit::WebImage*, int32_t) { }
};

} // namespace API

#endif // APIFindMatchesClient_h
