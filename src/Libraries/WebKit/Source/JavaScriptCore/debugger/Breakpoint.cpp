/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#include "config.h"
#include "Breakpoint.h"

#include "Debugger.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Breakpoint);

Breakpoint::Action::Action(Breakpoint::Action::Type type)
    : type(type)
{
}

Ref<Breakpoint> Breakpoint::create(BreakpointID id, const String& condition, ActionsVector&& actions, bool autoContinue, size_t ignoreCount)
{
    return adoptRef(*new Breakpoint(id, condition, WTFMove(actions), autoContinue, ignoreCount));
}

Breakpoint::Breakpoint(BreakpointID id, String condition, ActionsVector&& actions, bool autoContinue, size_t ignoreCount)
    : m_id(id)
    , m_condition(condition)
    , m_actions(WTFMove(actions))
    , m_autoContinue(autoContinue)
    , m_ignoreCount(ignoreCount)
{
}

bool Breakpoint::link(SourceID sourceID, unsigned lineNumber, unsigned columnNumber)
{
    ASSERT(!isLinked());
    ASSERT(!isResolved());

    m_sourceID = sourceID;
    m_lineNumber = lineNumber;
    m_columnNumber = columnNumber;
    return isLinked();
}

bool Breakpoint::resolve(unsigned lineNumber, unsigned columnNumber)
{
    ASSERT(isLinked());
    ASSERT(!isResolved());
    ASSERT(lineNumber >= m_lineNumber);
    ASSERT(columnNumber >= m_columnNumber || lineNumber > m_lineNumber);

    m_lineNumber = lineNumber;
    m_columnNumber = columnNumber;
    m_resolved = true;
    return isResolved();
}

bool Breakpoint::shouldPause(Debugger& debugger, JSGlobalObject* globalObject)
{
    if (!debugger.evaluateBreakpointCondition(*this, globalObject))
        return false;

    return ++m_hitCount > m_ignoreCount;
}

} // namespace JSC
