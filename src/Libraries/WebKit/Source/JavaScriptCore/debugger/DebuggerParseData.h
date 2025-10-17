/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

#include "ParserTokens.h"
#include <wtf/Function.h>
#include <wtf/Vector.h>

namespace JSC {

class SourceProvider;
class VM;

// The order of the constants here matters as it is used for
// sorting pause positions that have same offset.
enum class DebuggerPausePositionType { Invalid, Enter, Pause, Leave };
struct DebuggerPausePosition {
    DebuggerPausePositionType type { DebuggerPausePositionType::Invalid };
    JSTextPosition position;
};

class DebuggerPausePositions {
public:
    DebuggerPausePositions() { }
    ~DebuggerPausePositions() { }

    void appendPause(const JSTextPosition& position)
    {
        m_positions.append({ DebuggerPausePositionType::Pause, position });
    }

    void appendEntry(const JSTextPosition& position)
    {
        m_positions.append({ DebuggerPausePositionType::Enter, position });
    }

    void appendLeave(const JSTextPosition& position)
    {
        m_positions.append({ DebuggerPausePositionType::Leave, position });
    }

    void forEachBreakpointLocation(int startLine, int startColumn, int endLine, int endColumn, Function<void(const JSTextPosition&)>&&);

    std::optional<JSTextPosition> breakpointLocationForLineColumn(int line, int column);

    void sort();

private:
    using Positions = Vector<DebuggerPausePosition>;

    Positions::iterator firstPositionAfter(int line, int column);
    std::optional<JSTextPosition> breakpointLocationForLineColumn(int line, int column, Positions::iterator);

    Positions m_positions;
};


struct DebuggerParseData {
    DebuggerParseData() { }
    ~DebuggerParseData() { }

    DebuggerPausePositions pausePositions;
};

bool gatherDebuggerParseDataForSource(VM&, SourceProvider*, DebuggerParseData&);

} // namespace JSC
