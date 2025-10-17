/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
#include "config.h"

#include "IntersectionObserverEntry.h"

#include "Element.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IntersectionObserverEntry);

IntersectionObserverEntry::IntersectionObserverEntry(const Init& init)
    : m_time(init.time)
    , m_boundingClientRect(DOMRectReadOnly::fromRect(init.boundingClientRect))
    , m_intersectionRect(DOMRectReadOnly::fromRect(init.intersectionRect))
    , m_intersectionRatio(init.intersectionRatio)
    , m_target(init.target)
    , m_isIntersecting(init.isIntersecting)
{
    if (init.rootBounds)
        m_rootBounds = DOMRectReadOnly::fromRect(*init.rootBounds);
}

TextStream& operator<<(TextStream& ts, const IntersectionObserverEntry& entry)
{
    TextStream::GroupScope scope(ts);
    ts << "IntersectionObserverEntry " << &entry;
    ts.dumpProperty("time", entry.time());
    
    if (entry.rootBounds())
        ts.dumpProperty("rootBounds", entry.rootBounds()->toFloatRect());

    if (entry.boundingClientRect())
        ts.dumpProperty("boundingClientRect", entry.boundingClientRect()->toFloatRect());

    if (entry.intersectionRect())
        ts.dumpProperty("intersectionRect", entry.intersectionRect()->toFloatRect());

    ts.dumpProperty("isIntersecting", entry.isIntersecting());
    ts.dumpProperty("intersectionRatio", entry.intersectionRatio());

    return ts;
}

} // namespace WebCore
