/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#include "RenderBlock.h"
#include "RenderStyle.h"
#include <optional>
#include <wtf/HashSet.h>

namespace WebCore {

struct WordTrailingSpace {
    WordTrailingSpace(const RenderStyle& style, bool measuringWithTrailingWhitespaceEnabled = true)
        : m_style(style)
    {
        if (!measuringWithTrailingWhitespaceEnabled || !m_style.fontCascade().enableKerning())
            m_state = WordTrailingSpaceState::Initialized;
    }

    std::optional<float> width(SingleThreadWeakHashSet<const Font>& fallbackFonts)
    {
        if (m_state == WordTrailingSpaceState::Initialized)
            return m_width;

        auto& font = m_style.fontCascade();
        m_width = font.width(RenderBlock::constructTextRun(span(space), m_style), &fallbackFonts) + font.wordSpacing();
        m_state = WordTrailingSpaceState::Initialized;
        return m_width;
    }

private:
    enum class WordTrailingSpaceState { Uninitialized, Initialized };
    const RenderStyle& m_style;
    WordTrailingSpaceState m_state { WordTrailingSpaceState::Uninitialized };
    std::optional<float> m_width;
};

} // namespace WebCore
