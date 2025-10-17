/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class Element;

class IntersectionObserverEntry : public RefCounted<IntersectionObserverEntry> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IntersectionObserverEntry);
public:

    struct Init {
        double time;
        std::optional<DOMRectInit> rootBounds;
        DOMRectInit boundingClientRect;
        DOMRectInit intersectionRect;
        double intersectionRatio;
        RefPtr<Element> target;
        bool isIntersecting;
    };

    static Ref<IntersectionObserverEntry> create(const Init& init)
    {
        return adoptRef(*new IntersectionObserverEntry(init));
    }
    
    double time() const { return m_time; }
    DOMRectReadOnly* rootBounds() const { return m_rootBounds.get(); }
    DOMRectReadOnly* boundingClientRect() const { return m_boundingClientRect.get(); }
    DOMRectReadOnly* intersectionRect() const { return m_intersectionRect.get(); }
    Element* target() const { return m_target.get(); }

    bool isIntersecting() const { return m_isIntersecting; }
    double intersectionRatio() const { return m_intersectionRatio; }

private:
    IntersectionObserverEntry(const Init&);

    double m_time { 0 };
    RefPtr<DOMRectReadOnly> m_rootBounds;
    RefPtr<DOMRectReadOnly> m_boundingClientRect;
    RefPtr<DOMRectReadOnly> m_intersectionRect;
    double m_intersectionRatio { 0 };
    RefPtr<Element> m_target;
    bool m_isIntersecting { false };
};

TextStream& operator<<(TextStream&, const IntersectionObserverEntry&);

} // namespace WebCore
