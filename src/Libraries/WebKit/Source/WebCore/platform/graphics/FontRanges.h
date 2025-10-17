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

#include "Font.h"
#include <wtf/Vector.h>

namespace WebCore {

class FontAccessor;

enum class ExternalResourceDownloadPolicy : bool {
    Forbid,
    Allow
};

enum class IsGenericFontFamily : bool {
    No,
    Yes
};

class FontRanges {
public:
    struct Range {
        Range(char32_t from, char32_t to, Ref<FontAccessor>&& fontAccessor)
            : m_from(from)
            , m_to(to)
            , m_fontAccessor(WTFMove(fontAccessor))
        {
        }

        Range(const Range&) = default;
        Range(Range&&) = default;
        Range& operator=(const Range&) = delete;
        Range& operator=(Range&&) = default;

        char32_t from() const { return m_from; }
        char32_t to() const { return m_to; }
        WEBCORE_EXPORT const Font* font(ExternalResourceDownloadPolicy) const;
        const FontAccessor& fontAccessor() const { return m_fontAccessor; }

    private:
        char32_t m_from;
        char32_t m_to;
        Ref<FontAccessor> m_fontAccessor;
    };

    FontRanges() = default;
    explicit FontRanges(RefPtr<Font>&&);
    ~FontRanges();

    FontRanges(const FontRanges&) = default;
    FontRanges(FontRanges&& other, IsGenericFontFamily);
    FontRanges& operator=(FontRanges&&) = default;

    bool isNull() const { return m_ranges.isEmpty(); }

    void appendRange(Range&& range) { m_ranges.append(WTFMove(range)); }
    unsigned size() const { return m_ranges.size(); }
    const Range& rangeAt(unsigned i) const { return m_ranges[i]; }

    void shrinkToFit() { m_ranges.shrinkToFit(); }

    WEBCORE_EXPORT GlyphData glyphDataForCharacter(char32_t, ExternalResourceDownloadPolicy) const;
    WEBCORE_EXPORT const Font* fontForCharacter(char32_t) const;
    WEBCORE_EXPORT const Font& fontForFirstRange() const;
    bool isLoading() const;
    bool isGenericFontFamily() const { return m_isGenericFontFamily == IsGenericFontFamily::Yes; }

private:
    Vector<Range, 1> m_ranges;
    IsGenericFontFamily m_isGenericFontFamily { IsGenericFontFamily::No };
};

}
