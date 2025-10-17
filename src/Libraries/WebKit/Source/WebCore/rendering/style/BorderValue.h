/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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

#include "RenderStyleConstants.h"
#include "StyleColor.h"

namespace WebCore {

class RenderStyle;

class BorderValue {
friend class RenderStyle;
public:
    BorderValue();

    bool nonZero() const
    {
        return width() && style() != BorderStyle::None;
    }

    bool isTransparent() const;

    bool isVisible() const;

    friend bool operator==(const BorderValue&, const BorderValue&) = default;

    void setColor(const Style::Color& color)
    {
        m_color = color;
    }

    const Style::Color& color() const { return m_color; }

    float width() const { return m_width; }
    BorderStyle style() const { return static_cast<BorderStyle>(m_style); }

protected:
    Style::Color m_color;

    float m_width { 3 };

    unsigned m_style : 4; // BorderStyle

    // This is only used by OutlineValue but moved here to keep the bits packed.
    unsigned m_isAuto : 1; // OutlineIsAuto
};

} // namespace WebCore
