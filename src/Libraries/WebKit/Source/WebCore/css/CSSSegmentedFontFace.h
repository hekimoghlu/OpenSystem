/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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

#include "CSSFontFace.h"
#include "FontCache.h"
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class CSSFontSelector;
class FontCreationContext;
class FontDescription;
class FontPaletteValues;
class FontRanges;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(CSSSegmentedFontFace);
class CSSSegmentedFontFace final : public RefCounted<CSSSegmentedFontFace>, public CSSFontFaceClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(CSSSegmentedFontFace);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<CSSSegmentedFontFace> create()
    {
        return adoptRef(*new CSSSegmentedFontFace());
    }
    ~CSSSegmentedFontFace();

    void appendFontFace(Ref<CSSFontFace>&&);

    FontRanges fontRanges(const FontDescription&, const FontPaletteValues&, RefPtr<FontFeatureValues>);

    Vector<Ref<CSSFontFace>, 1>& constituentFaces() { return m_fontFaces; }

private:
    CSSSegmentedFontFace();
    void fontLoaded(CSSFontFace&) final;

    // FIXME: Add support for font-feature-values in the key for this cache.
    UncheckedKeyHashMap<std::tuple<FontDescriptionKey, FontPaletteValues>, FontRanges> m_cache;
    Vector<Ref<CSSFontFace>, 1> m_fontFaces;
};

} // namespace WebCore
