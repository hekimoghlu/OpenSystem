/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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

#include "BorderValue.h"
#include "GapLength.h"
#include "RenderStyleConstants.h"
#include <wtf/RefCounted.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

// CSS3 Multi Column Layout

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleMultiColData);
class StyleMultiColData : public RefCounted<StyleMultiColData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleMultiColData);
public:
    static Ref<StyleMultiColData> create() { return adoptRef(*new StyleMultiColData); }
    Ref<StyleMultiColData> copy() const;
    
    bool operator==(const StyleMultiColData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleMultiColData&) const;
#endif

    unsigned short ruleWidth() const
    {
        if (rule.style() == BorderStyle::None || rule.style() == BorderStyle::Hidden)
            return 0; 
        return rule.width();
    }

    float width { 0 };
    unsigned short count;
    BorderValue rule;
    Style::Color visitedLinkColumnRuleColor;

    bool autoWidth : 1;
    bool autoCount : 1;
    unsigned fill : 1; // ColumnFill
    unsigned columnSpan : 1; // ColumnSpan
    unsigned axis : 2; // ColumnAxis
    unsigned progression : 2; // ColumnProgression

private:
    StyleMultiColData();
    StyleMultiColData(const StyleMultiColData&);
};

} // namespace WebCore
