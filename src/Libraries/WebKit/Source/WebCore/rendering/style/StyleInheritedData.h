/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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

#include "FontCascade.h"
#include "Length.h"
#include "StyleColor.h"
#include "StyleFontData.h"
#include <wtf/DataRef.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleInheritedData);
class StyleInheritedData : public RefCounted<StyleInheritedData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleInheritedData);
public:
    static Ref<StyleInheritedData> create() { return adoptRef(*new StyleInheritedData); }
    Ref<StyleInheritedData> copy() const;

    bool operator==(const StyleInheritedData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleInheritedData&) const;
#endif

    bool fastPathInheritedEqual(const StyleInheritedData&) const;
    bool nonFastPathInheritedEqual(const StyleInheritedData&) const;
    void fastPathInheritFrom(const StyleInheritedData&);

    float horizontalBorderSpacing;
    float verticalBorderSpacing;

    Length lineHeight;
#if ENABLE(TEXT_AUTOSIZING)
    Length specifiedLineHeight;
#endif

    DataRef<StyleFontData> fontData;
    Color color;
    Color visitedLinkColor;

private:
    StyleInheritedData();
    StyleInheritedData(const StyleInheritedData&);
    void operator=(const StyleInheritedData&) = delete;
};

} // namespace WebCore
