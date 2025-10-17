/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include <wtf/RefCounted.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

using SharedStringHash = uint32_t;
class Page;

class VisitedLinkStore : public RefCounted<VisitedLinkStore> {
public:
    WEBCORE_EXPORT VisitedLinkStore();
    WEBCORE_EXPORT virtual ~VisitedLinkStore();

    // FIXME: These two members should only take the link hash.
    virtual bool isLinkVisited(Page&, SharedStringHash, const URL& baseURL, const AtomString& attributeURL) = 0;
    virtual void addVisitedLink(Page&, SharedStringHash) = 0;

    void addPage(Page&);
    void removePage(Page&);

    WEBCORE_EXPORT void invalidateStylesForAllLinks();
    WEBCORE_EXPORT void invalidateStylesForLink(SharedStringHash);

private:
    WeakHashSet<Page> m_pages;
};

} // namespace WebCore
