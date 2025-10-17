/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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

#include <wtf/Vector.h>

namespace WebCore {

class LegacyInlineIterator;
class RenderBoxModelObject;
class RenderText;

struct BidiRun;
struct BidiIsolatedRun;

template <class Iterator, class Run> class BidiResolver;
template <class Iterator, class Run, class IsolateRun> class BidiResolverWithIsolate;
template <class Iterator> class WhitespaceCollapsingState;
typedef BidiResolverWithIsolate<LegacyInlineIterator, BidiRun, BidiIsolatedRun> InlineBidiResolver;
typedef WhitespaceCollapsingState<LegacyInlineIterator> LineWhitespaceCollapsingState;

class TrailingObjects {
public:
    void setTrailingWhitespace(RenderText& whitespace) { m_whitespace = &whitespace; }
    void clear()
    {
        m_whitespace = { };
        m_boxes.shrink(0); // Use shrink(0) instead of clear() to retain our capacity.
    }

    void appendBoxIfNeeded(RenderBoxModelObject& box)
    {
        if (m_whitespace)
            m_boxes.append(box);
    }

    enum class CollapseFirstSpace : bool { No, Yes };
    void updateWhitespaceCollapsingTransitionsForTrailingBoxes(LineWhitespaceCollapsingState&, const LegacyInlineIterator& lBreak, CollapseFirstSpace);

private:
    RenderText* m_whitespace { nullptr };
    Vector<std::reference_wrapper<RenderBoxModelObject>, 4> m_boxes;
};

} // namespace WebCore
