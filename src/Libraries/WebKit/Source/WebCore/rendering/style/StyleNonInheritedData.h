/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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

#include <wtf/DataRef.h>
#include <wtf/RefCounted.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class StyleBoxData;
class StyleBackgroundData;
class StyleSurroundData;
class StyleMiscNonInheritedData;
class StyleRareNonInheritedData;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleNonInheritedData);
class StyleNonInheritedData : public RefCounted<StyleNonInheritedData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleNonInheritedData);
public:
    static Ref<StyleNonInheritedData> create();
    Ref<StyleNonInheritedData> copy() const;

    bool operator==(const StyleNonInheritedData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleNonInheritedData&) const;
#endif

    DataRef<StyleBoxData> boxData;
    DataRef<StyleBackgroundData> backgroundData;
    DataRef<StyleSurroundData> surroundData;
    DataRef<StyleMiscNonInheritedData> miscData;
    DataRef<StyleRareNonInheritedData> rareData;

private:
    StyleNonInheritedData();
    StyleNonInheritedData(const StyleNonInheritedData&);
};

} // namespace WebCore
