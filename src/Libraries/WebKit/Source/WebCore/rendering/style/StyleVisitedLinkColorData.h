/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

#include "StyleColor.h"

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleVisitedLinkColorData);
class StyleVisitedLinkColorData : public RefCounted<StyleVisitedLinkColorData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleVisitedLinkColorData);
public:
    static Ref<StyleVisitedLinkColorData> create() { return adoptRef(*new StyleVisitedLinkColorData); }
    Ref<StyleVisitedLinkColorData> copy() const;
    ~StyleVisitedLinkColorData();

    bool operator==(const StyleVisitedLinkColorData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleVisitedLinkColorData&) const;
#endif

    Style::Color background;
    Style::Color borderLeft;
    Style::Color borderRight;
    Style::Color borderTop;
    Style::Color borderBottom;
    Style::Color textDecoration;
    Style::Color outline;

private:
    StyleVisitedLinkColorData();
    StyleVisitedLinkColorData(const StyleVisitedLinkColorData&);
};

} // namespace WebCore
