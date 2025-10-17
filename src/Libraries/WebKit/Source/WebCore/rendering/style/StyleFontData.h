/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#include <wtf/DataRef.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class StyleFontData : public RefCounted<StyleFontData> {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static Ref<StyleFontData> create() { return adoptRef(*new StyleFontData); }
    Ref<StyleFontData> copy() const;

    bool operator==(const StyleFontData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleFontData&) const;
#endif

    FontCascade fontCascade;

private:
    StyleFontData();
    StyleFontData(const StyleFontData&);
    void operator=(const StyleFontData&) = delete;
};

} // namespace WebCore
