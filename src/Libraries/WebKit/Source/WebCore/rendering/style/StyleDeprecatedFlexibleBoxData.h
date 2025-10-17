/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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

#include <wtf/RefCounted.h>
#include <wtf/Ref.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleDeprecatedFlexibleBoxData);
class StyleDeprecatedFlexibleBoxData : public RefCounted<StyleDeprecatedFlexibleBoxData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleDeprecatedFlexibleBoxData);
public:
    static Ref<StyleDeprecatedFlexibleBoxData> create() { return adoptRef(*new StyleDeprecatedFlexibleBoxData); }
    Ref<StyleDeprecatedFlexibleBoxData> copy() const;

    bool operator==(const StyleDeprecatedFlexibleBoxData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleDeprecatedFlexibleBoxData&) const;
#endif

    float flex;
    unsigned flexGroup;
    unsigned ordinalGroup;

    unsigned align : 3; // BoxAlignment
    unsigned pack: 2; // BoxPack
    unsigned orient: 1; // BoxOrient
    unsigned lines : 1; // BoxLines

private:
    StyleDeprecatedFlexibleBoxData();
    StyleDeprecatedFlexibleBoxData(const StyleDeprecatedFlexibleBoxData&);
};

} // namespace WebCore
