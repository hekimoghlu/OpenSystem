/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#include "JavaScriptCallFrame.h"

namespace Inspector {

using namespace JSC;

JavaScriptCallFrame::JavaScriptCallFrame(Ref<DebuggerCallFrame>&& debuggerCallFrame)
    : m_debuggerCallFrame(WTFMove(debuggerCallFrame))
{
}

JavaScriptCallFrame* JavaScriptCallFrame::caller()
{
    if (m_caller)
        return m_caller.get();

    RefPtr<DebuggerCallFrame> debuggerCallerFrame = m_debuggerCallFrame->callerFrame();
    if (!debuggerCallerFrame)
        return nullptr;

    m_caller = create(debuggerCallerFrame.releaseNonNull());
    return m_caller.get();
}

} // namespace Inspector

