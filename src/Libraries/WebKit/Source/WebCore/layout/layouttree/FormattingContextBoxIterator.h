/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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

#include "LayoutDescendantIterator.h"

namespace WebCore {
namespace Layout {

class FormattingContextBoxIterator : public LayoutDescendantIterator<Box> {
public:
    FormattingContextBoxIterator(const ElementBox& root)
        : LayoutDescendantIterator<Layout::Box>(root)
    {
    }

    enum FirstTag { First };
    FormattingContextBoxIterator(const ElementBox& root, FirstTag)
        : LayoutDescendantIterator<Box>(root, Traversal::firstWithin<Box>(root))
    {
    }

    FormattingContextBoxIterator& operator++()
    {
        if (get().establishesFormattingContext())
            traverseNextSkippingChildren();
        else
            traverseNext();
        return *this;
    }
};

class FormattingContextBoxIteratorAdapter {
public:
    FormattingContextBoxIteratorAdapter(const ElementBox& root)
        : m_root(root)
    {
    }
    FormattingContextBoxIterator begin() { return { m_root, FormattingContextBoxIterator::First }; }
    FormattingContextBoxIterator end() { return { m_root }; }

private:
    const ElementBox& m_root;
};

inline FormattingContextBoxIteratorAdapter formattingContextBoxes(const ElementBox& root)
{
    ASSERT(root.establishesFormattingContext());
    return { root };
}


}
}
