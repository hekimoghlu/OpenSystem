/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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

#include "DebuggerPrimitives.h"
#include <wtf/Noncopyable.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class Debugger;
class JSGlobalObject;

class Breakpoint : public RefCounted<Breakpoint> {
    WTF_MAKE_NONCOPYABLE(Breakpoint);
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(Breakpoint, JS_EXPORT_PRIVATE);
public:
    struct Action {
        enum class Type : uint8_t {
            Log,
            Evaluate,
            Sound,
            Probe,
        };

        Action(Type);

        Type type;
        String data;
        BreakpointActionID id { noBreakpointActionID };
        bool emulateUserGesture { false };
    };

    using ActionsVector = Vector<Action>;

    JS_EXPORT_PRIVATE static Ref<Breakpoint> create(BreakpointID, const String& condition = nullString(), ActionsVector&& = { }, bool autoContinue = false, size_t ignoreCount = 0);

    BreakpointID id() const { return m_id; }

    SourceID sourceID() const { return m_sourceID; }
    unsigned lineNumber() const { return m_lineNumber; }
    unsigned columnNumber() const { return m_columnNumber; }

    const String& condition() const { return m_condition; }
    const ActionsVector& actions() const { return m_actions; }
    bool isAutoContinue() const { return m_autoContinue; }

    void resetHitCount() { m_hitCount = 0; }

    // Associates this breakpoint with a position in a specific source code.
    bool link(SourceID, unsigned lineNumber, unsigned columnNumber);
    bool isLinked() const { return m_sourceID != noSourceID; }

    // Adjust the previously associated position to the next pause opportunity.
    bool resolve(unsigned lineNumber, unsigned columnNumber);
    bool isResolved() const { return m_resolved; }

    bool shouldPause(Debugger&, JSGlobalObject*);

private:
    Breakpoint(BreakpointID, String condition = nullString(), ActionsVector&& = { }, bool autoContinue = false, size_t ignoreCount = 0);

    BreakpointID m_id { noBreakpointID };

    SourceID m_sourceID { noSourceID };

    // FIXME: <https://webkit.org/b/162771> Web Inspector: Adopt TextPosition in Inspector to avoid oneBasedInt/zeroBasedInt ambiguity
    unsigned m_lineNumber { 0 };
    unsigned m_columnNumber { 0 };

    bool m_resolved { false };

    String m_condition;
    ActionsVector m_actions;
    bool m_autoContinue { false };
    size_t m_ignoreCount { 0 };
    size_t m_hitCount { 0 };
};

using BreakpointsVector = Vector<Ref<JSC::Breakpoint>>;

} // namespace JSC
