/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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

#include "InlineIteratorBox.h"
#include "InlineIteratorBoxModernPathInlines.h"

namespace WebCore {
namespace InlineIterator {

inline float Box::logicalBottom() const { return logicalRectIgnoringInlineDirection().maxY(); }
inline float Box::logicalHeight() const { return logicalRectIgnoringInlineDirection().height(); }
inline float Box::logicalLeftIgnoringInlineDirection() const { return logicalRectIgnoringInlineDirection().x(); }
inline float Box::logicalRightIgnoringInlineDirection() const { return logicalRectIgnoringInlineDirection().maxX(); }
inline float Box::logicalTop() const { return logicalRectIgnoringInlineDirection().y(); }
inline float Box::logicalWidth() const { return logicalRectIgnoringInlineDirection().width(); }

inline float Box::logicalLeft() const { return isHorizontal() ? visualRect().x() : visualRect().y(); }
inline float Box::logicalRight() const { return logicalLeft() + logicalWidth(); }

inline bool Box::isHorizontal() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.isHorizontal();
    });
}

inline FloatRect Box::logicalRectIgnoringInlineDirection() const
{
    auto rect = this->visualRectIgnoringBlockDirection();
    return isHorizontal() ? rect : rect.transposedRect();
}

// Coordinate-relative left/right
inline LeafBoxIterator Box::nextLogicalRightwardOnLine() const
{
    return writingMode().isLogicalLeftLineLeft()
        ? nextLineRightwardOnLine() : nextLineLeftwardOnLine();
}

inline LeafBoxIterator Box::nextLogicalLeftwardOnLine() const
{
    return writingMode().isLogicalLeftLineLeft()
        ? nextLineLeftwardOnLine() : nextLineRightwardOnLine();
}

inline LeafBoxIterator Box::nextLogicalRightwardOnLineIgnoringLineBreak() const
{
    return writingMode().isLogicalLeftLineLeft()
        ? nextLineRightwardOnLineIgnoringLineBreak() : nextLineLeftwardOnLineIgnoringLineBreak();
}

inline LeafBoxIterator Box::nextLogicalLeftwardOnLineIgnoringLineBreak() const
{
    return writingMode().isLogicalLeftLineLeft()
        ? nextLineLeftwardOnLineIgnoringLineBreak() : nextLineRightwardOnLineIgnoringLineBreak();
}

inline LeafBoxIterator& LeafBoxIterator::traverseLogicalRightwardOnLine()
{
    return m_box.writingMode().isLogicalLeftLineLeft()
        ? traverseLineRightwardOnLine()
        : traverseLineLeftwardOnLine();
}

inline LeafBoxIterator& LeafBoxIterator::traverseLogicalLeftwardOnLine()
{
    return m_box.writingMode().isLogicalLeftLineLeft()
        ? traverseLineLeftwardOnLine()
        : traverseLineRightwardOnLine();
}

inline LeafBoxIterator& LeafBoxIterator::traverseLogicalRightwardOnLineIgnoringLineBreak()
{
    return m_box.writingMode().isLogicalLeftLineLeft()
        ? traverseLineRightwardOnLineIgnoringLineBreak()
        : traverseLineLeftwardOnLineIgnoringLineBreak();
}

inline LeafBoxIterator& LeafBoxIterator::traverseLogicalLeftwardOnLineIgnoringLineBreak()
{
    return m_box.writingMode().isLogicalLeftLineLeft()
        ? traverseLineLeftwardOnLineIgnoringLineBreak()
        : traverseLineRightwardOnLineIgnoringLineBreak();
}

}
}
