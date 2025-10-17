/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include "LegacyInlineIterator.h"
#include "LineInfo.h"
#include <wtf/Vector.h>

namespace WebCore {

class RenderText;
class TextLayout;

struct RenderTextInfo {
    RenderText* text { nullptr };
    std::unique_ptr<TextLayout, TextLayoutDeleter> layout;
    CachedLineBreakIteratorFactory lineBreakIteratorFactory;
    const FontCascade* font { nullptr };
};

class LineBreaker {
public:
    friend class BreakingContext;

    explicit LineBreaker(RenderBlockFlow& block)
        : m_block(block)
    {
    }

    LegacyInlineIterator nextLineBreak(InlineBidiResolver&, LineInfo&, RenderTextInfo&);

private:
    void skipTrailingWhitespace(LegacyInlineIterator&, const LineInfo&);
    void skipLeadingWhitespace(InlineBidiResolver&, LineInfo&);

    RenderBlockFlow& m_block;
};

} // namespace WebCore
