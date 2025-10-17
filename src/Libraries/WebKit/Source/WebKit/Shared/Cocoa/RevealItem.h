/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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

#if ENABLE(REVEAL)

#import <wtf/RetainPtr.h>
#import <wtf/text/WTFString.h>

OBJC_CLASS RVItem;

typedef struct _NSRange NSRange;
typedef unsigned long NSUInteger;

namespace WebKit {

struct RevealItemRange {
    RevealItemRange() = default;
    RevealItemRange(NSRange);
    RevealItemRange(NSUInteger loc, NSUInteger len)
        : location(loc)
        , length(len)
    {
    }

    NSUInteger location { 0 };
    NSUInteger length { 0 };
};

class RevealItem {
public:
    RevealItem() = default;
    RevealItem(const String& text, RevealItemRange selectedRange);

    const String& text() const { return m_text; }
    const RevealItemRange& selectedRange() const { return m_selectedRange; }
    NSRange highlightRange() const;

    RVItem *item() const;

private:
    String m_text;
    RevealItemRange m_selectedRange;

    mutable RetainPtr<RVItem> m_item;
};

}

#endif // ENABLE(REVEAL)
