/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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

#include "LineWidth.h"

namespace WebCore {

class RenderBlock;

class LineInfo {
public:
    LineInfo()
        : m_isFirstLine(true)
        , m_isLastLine(false)
        , m_isEmpty(true)
        , m_runsFromLeadingWhitespace(0)
    { }

    bool isFirstLine() const { return m_isFirstLine; }
    bool isLastLine() const { return m_isLastLine; }
    bool isEmpty() const { return m_isEmpty; }
    unsigned runsFromLeadingWhitespace() const { return m_runsFromLeadingWhitespace; }
    void resetRunsFromLeadingWhitespace() { m_runsFromLeadingWhitespace = 0; }
    void incrementRunsFromLeadingWhitespace() { m_runsFromLeadingWhitespace++; }

    void setFirstLine(bool firstLine) { m_isFirstLine = firstLine; }
    void setLastLine(bool lastLine) { m_isLastLine = lastLine; }
    void setEmpty(bool);

private:
    bool m_isFirstLine;
    bool m_isLastLine;
    bool m_isEmpty;
    unsigned m_runsFromLeadingWhitespace;
};

} // namespace WebCore
