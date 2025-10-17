/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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

#include "FillLayer.h"
#include "OutlineValue.h"
#include "StyleColor.h"
#include <wtf/DataRef.h>
#include <wtf/RefCounted.h>
#include <wtf/Ref.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleBackgroundData);
class StyleBackgroundData : public RefCounted<StyleBackgroundData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleBackgroundData);
public:
    static Ref<StyleBackgroundData> create() { return adoptRef(*new StyleBackgroundData); }
    Ref<StyleBackgroundData> copy() const;

    bool operator==(const StyleBackgroundData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleBackgroundData&) const;
#endif

    bool isEquivalentForPainting(const StyleBackgroundData&, bool currentColorDiffers) const;

    DataRef<FillLayer> background;
    Style::Color color;
    OutlineValue outline;

    void dump(TextStream&, DumpStyleValues = DumpStyleValues::All) const;

private:
    StyleBackgroundData();
    StyleBackgroundData(const StyleBackgroundData&);
};

WTF::TextStream& operator<<(WTF::TextStream&, const StyleBackgroundData&);

} // namespace WebCore
