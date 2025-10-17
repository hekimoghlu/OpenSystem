/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
#include "RegionContext.h"

#include "Path.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RegionContext);

void RegionContext::pushTransform(const AffineTransform& transform)
{
    if (m_transformStack.isEmpty())
        m_transformStack.append(transform);
    else
        m_transformStack.append(m_transformStack.last() * transform);
}

void RegionContext::popTransform()
{
    if (m_transformStack.isEmpty()) {
        ASSERT_NOT_REACHED();
        return;
    }
    m_transformStack.removeLast();
}

void RegionContext::pushClip(const IntRect& clipRect)
{
    auto transformedClip = m_transformStack.isEmpty() ? clipRect : m_transformStack.last().mapRect(clipRect);

    if (m_clipStack.isEmpty())
        m_clipStack.append(transformedClip);
    else
        m_clipStack.append(intersection(m_clipStack.last(), transformedClip));
}

void RegionContext::pushClip(const Path& path)
{
    // FIXME: Approximate paths better.
    auto pathBounds = enclosingIntRect(path.boundingRect());
    pushClip(pathBounds);
}

void RegionContext::popClip()
{
    if (m_clipStack.isEmpty()) {
        ASSERT_NOT_REACHED();
        return;
    }
    m_clipStack.removeLast();
}

} // namespace WebCore
