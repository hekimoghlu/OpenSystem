/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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

#include "DOMRectReadOnly.h"
#include "Element.h"
#include "FloatRect.h"
#include "ResizeObserverSize.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class Element;
class ResizeObserverSize;

class ResizeObserverEntry : public RefCounted<ResizeObserverEntry> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ResizeObserverEntry);
public:
    static Ref<ResizeObserverEntry> create(Element* target, const FloatRect& contentRect, FloatSize borderBoxSize, FloatSize contentBoxSize)
    {
        return adoptRef(*new ResizeObserverEntry(target, contentRect, borderBoxSize, contentBoxSize));
    }

    Element* target() const { return m_target.get(); }
    DOMRectReadOnly* contentRect() const { return m_contentRect.ptr(); }
    
    const Vector<Ref<ResizeObserverSize>>& borderBoxSize() const { return m_borderBoxSizes; }
    const Vector<Ref<ResizeObserverSize>>& contentBoxSize() const { return m_contentBoxSizes; }

private:
    ResizeObserverEntry(Element* target, const FloatRect& contentRect, FloatSize borderBoxSize, FloatSize contentBoxSize)
        : m_target(target)
        , m_contentRect(DOMRectReadOnly::create(contentRect.x(), contentRect.y(), contentRect.width(), contentRect.height()))
        , m_borderBoxSizes({ ResizeObserverSize::create(borderBoxSize.width(), borderBoxSize.height()) })
        , m_contentBoxSizes({ ResizeObserverSize::create(contentBoxSize.width(), contentBoxSize.height()) })
    {
    }

    RefPtr<Element> m_target;
    Ref<DOMRectReadOnly> m_contentRect;
    // The spec is designed to allow mulitple boxes for multicol scenarios, but for now these vectors only ever contain one entry.
    Vector<Ref<ResizeObserverSize>> m_borderBoxSizes;
    Vector<Ref<ResizeObserverSize>> m_contentBoxSizes;
};

} // namespace WebCore
