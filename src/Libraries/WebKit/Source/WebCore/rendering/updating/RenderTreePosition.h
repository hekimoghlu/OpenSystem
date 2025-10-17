/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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

#include "RenderElement.h"

namespace WebCore {

class RenderTreePosition {
public:
    explicit RenderTreePosition(RenderElement& parent)
        : m_parent(parent)
    {
    }
    RenderTreePosition(RenderElement& parent, RenderObject* nextSibling)
        : m_parent(parent)
        , m_nextSibling(nextSibling)
        , m_hasValidNextSibling(true)
    {
    }

    RenderElement& parent() const { return m_parent.get(); }
    RenderObject* nextSibling() const { ASSERT(m_hasValidNextSibling); return m_nextSibling.get(); }

    void computeNextSibling(const Node&);
    void moveToLastChild();
    void invalidateNextSibling() { m_hasValidNextSibling = false; }
    void invalidateNextSibling(const RenderObject&);

    RenderObject* nextSiblingRenderer(const Node&) const;

private:
    CheckedRef<RenderElement> m_parent;
    SingleThreadWeakPtr<RenderObject> m_nextSibling;
    bool m_hasValidNextSibling { false };
#if ASSERT_ENABLED
    unsigned m_assertionLimitCounter { 0 };
#endif
};

inline void RenderTreePosition::moveToLastChild()
{
    m_nextSibling = nullptr;
    m_hasValidNextSibling = true;
}

} // namespace WebCore
