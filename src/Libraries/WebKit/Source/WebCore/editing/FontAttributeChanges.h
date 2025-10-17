/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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

#include "Color.h"
#include "FontShadow.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/Forward.h>

namespace IPC {
template<typename T, typename> struct ArgumentCoder;
}

namespace WebCore {

class EditingStyle;
class MutableStyleProperties;

enum class EditAction : uint8_t;
enum class VerticalAlignChange : uint8_t { Superscript, Baseline, Subscript };

class FontChanges {
public:
    FontChanges() = default;
    WEBCORE_EXPORT FontChanges(String&& fontName, String&& fontFamily, std::optional<double>&& fontSize, std::optional<double>&& fontSizeDelta, std::optional<bool>&& bold, std::optional<bool>&& italic);
    void setFontName(const String& fontName) { m_fontName = fontName; }
    void setFontFamily(const String& fontFamily) { m_fontFamily = fontFamily; }
    void setFontSize(double fontSize) { m_fontSize = fontSize; }
    void setFontSizeDelta(double fontSizeDelta) { m_fontSizeDelta = fontSizeDelta; }
    void setBold(bool bold) { m_bold = bold; }
    void setItalic(bool italic) { m_italic = italic; }

    bool isEmpty() const
    {
        return !m_fontName && !m_fontFamily && !m_fontSize && !m_fontSizeDelta && !m_bold && !m_italic;
    }

    WEBCORE_EXPORT Ref<EditingStyle> createEditingStyle() const;
    Ref<MutableStyleProperties> createStyleProperties() const;

private:
    friend struct IPC::ArgumentCoder<FontChanges, void>;
    const String& platformFontFamilyNameForCSS() const;

    String m_fontName;
    String m_fontFamily;
    std::optional<double> m_fontSize;
    std::optional<double> m_fontSizeDelta;
    std::optional<bool> m_bold;
    std::optional<bool> m_italic;
};

class FontAttributeChanges {
public:
    FontAttributeChanges() = default;
    WEBCORE_EXPORT FontAttributeChanges(std::optional<VerticalAlignChange>&&, std::optional<Color>&& backgroundColor, std::optional<Color>&& foregroundColor, std::optional<FontShadow>&&, std::optional<bool>&& strikeThrough, std::optional<bool>&& underline, FontChanges&&);

    void setVerticalAlign(VerticalAlignChange align) { m_verticalAlign = align; }
    void setBackgroundColor(const Color& color) { m_backgroundColor = color; }
    void setForegroundColor(const Color& color) { m_foregroundColor = color; }
    void setShadow(const FontShadow& shadow) { m_shadow = shadow; }
    void setStrikeThrough(bool strikeThrough) { m_strikeThrough = strikeThrough; }
    void setUnderline(bool underline) { m_underline = underline; }
    void setFontChanges(const FontChanges& fontChanges) { m_fontChanges = fontChanges; }

    WEBCORE_EXPORT Ref<EditingStyle> createEditingStyle() const;
    WEBCORE_EXPORT EditAction editAction() const;

private:
    friend struct IPC::ArgumentCoder<FontAttributeChanges, void>;
    std::optional<VerticalAlignChange> m_verticalAlign;
    std::optional<Color> m_backgroundColor;
    std::optional<Color> m_foregroundColor;
    std::optional<FontShadow> m_shadow;
    std::optional<bool> m_strikeThrough;
    std::optional<bool> m_underline;
    FontChanges m_fontChanges;
};

} // namespace WebCore
