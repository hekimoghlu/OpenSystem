/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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

#include "BorderData.h"
#include "LengthBox.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleSurroundData);
class StyleSurroundData : public RefCounted<StyleSurroundData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleSurroundData);
public:
    static Ref<StyleSurroundData> create() { return adoptRef(*new StyleSurroundData); }
    Ref<StyleSurroundData> copy() const;
    
    bool operator==(const StyleSurroundData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleSurroundData&) const;
#endif

    // Here instead of in BorderData to pack up against the refcount.
    bool hasExplicitlySetBorderBottomLeftRadius : 1;
    bool hasExplicitlySetBorderBottomRightRadius : 1;
    bool hasExplicitlySetBorderTopLeftRadius : 1;
    bool hasExplicitlySetBorderTopRightRadius : 1;

    LengthBox offset;
    LengthBox margin;
    LengthBox padding;
    BorderData border;
    
private:
    StyleSurroundData();
    StyleSurroundData(const StyleSurroundData&);    
};

} // namespace WebCore
