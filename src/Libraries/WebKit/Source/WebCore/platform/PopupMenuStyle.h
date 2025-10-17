/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
#include "FontCascade.h"
#include "Length.h"

namespace WebCore {

class PopupMenuStyle {
public:
    enum PopupMenuType { SelectPopup, AutofillPopup };
    enum BackgroundColorType { DefaultBackgroundColor, CustomBackgroundColor };
    enum class Size : uint8_t {
        Normal,
        Small,
        Mini,
        Large,
    };

    PopupMenuStyle(const Color& foreground, const Color& background, const FontCascade& font, bool visible, bool isDisplayNone, bool hasDefaultAppearance, Length textIndent, TextDirection textDirection, bool hasTextDirectionOverride, BackgroundColorType backgroundColorType = DefaultBackgroundColor, PopupMenuType menuType = SelectPopup, Size menuSize = Size::Normal)
        : m_foregroundColor(foreground)
        , m_backgroundColor(background)
        , m_font(font)
        , m_visible(visible)
        , m_isDisplayNone(isDisplayNone)
        , m_hasDefaultAppearance(hasDefaultAppearance)
        , m_textIndent(textIndent)
        , m_textDirection(textDirection)
        , m_hasTextDirectionOverride(hasTextDirectionOverride)
        , m_backgroundColorType(backgroundColorType)
        , m_menuType(menuType)
        , m_menuSize(menuSize)
    {
    }

    const Color& foregroundColor() const { return m_foregroundColor; }
    const Color& backgroundColor() const { return m_backgroundColor; }
    const FontCascade& font() const { return m_font; }
    bool isVisible() const { return m_visible; }
    bool isDisplayNone() const { return m_isDisplayNone; }
    bool hasDefaultAppearance() const { return m_hasDefaultAppearance; }
    Length textIndent() const { return m_textIndent; }
    TextDirection textDirection() const { return m_textDirection; }
    bool hasTextDirectionOverride() const { return m_hasTextDirectionOverride; }
    BackgroundColorType backgroundColorType() const { return m_backgroundColorType; }
    PopupMenuType menuType() const { return m_menuType; }
    Size menuSize() const { return m_menuSize; }

private:
    Color m_foregroundColor;
    Color m_backgroundColor;
    FontCascade m_font;
    bool m_visible;
    bool m_isDisplayNone;
    bool m_hasDefaultAppearance;
    Length m_textIndent;
    TextDirection m_textDirection;
    bool m_hasTextDirectionOverride;
    BackgroundColorType m_backgroundColorType;
    PopupMenuType m_menuType;
    Size m_menuSize;
};

} // namespace WebCore
