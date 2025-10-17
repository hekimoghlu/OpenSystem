/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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

#include "FloatQuad.h"
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class DOMRect;

class DOMRectList : public RefCounted<DOMRectList> {
public:
    static Ref<DOMRectList> create(const Vector<FloatQuad>& quads) { return adoptRef(*new DOMRectList(quads)); }
    static Ref<DOMRectList> create(const Vector<FloatRect>& rects) { return adoptRef(*new DOMRectList(rects)); }
    static Ref<DOMRectList> create() { return adoptRef(*new DOMRectList()); }
    WEBCORE_EXPORT ~DOMRectList();

    unsigned length() const { return m_items.size(); }
    DOMRect* item(unsigned index) { return index < m_items.size() ? m_items[index].ptr() : nullptr; }
    bool isSupportedPropertyIndex(unsigned index) const { return index < m_items.size(); }

private:
    WEBCORE_EXPORT explicit DOMRectList(const Vector<FloatQuad>& quads);
    WEBCORE_EXPORT explicit DOMRectList(const Vector<FloatRect>& rects);
    DOMRectList() = default;

    Vector<Ref<DOMRect>> m_items;
};

}
