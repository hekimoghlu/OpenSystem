/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include "ScriptCallStack.h"

namespace Inspector {

Ref<ScriptCallStack> ScriptCallStack::create()
{
    return adoptRef(*new ScriptCallStack);
}

Ref<ScriptCallStack> ScriptCallStack::create(Vector<ScriptCallFrame>&& frames, bool truncated, AsyncStackTrace* parentStackTrace)
{
    return adoptRef(*new ScriptCallStack(WTFMove(frames), truncated, parentStackTrace));
}

ScriptCallStack::ScriptCallStack() = default;

ScriptCallStack::ScriptCallStack(Vector<ScriptCallFrame>&& frames, bool truncated, AsyncStackTrace* parentStackTrace)
    : m_frames(WTFMove(frames))
    , m_truncated(truncated)
    , m_parentStackTrace(parentStackTrace)
{
    ASSERT(m_frames.size() <= maxCallStackSizeToCapture);
}

ScriptCallStack::~ScriptCallStack() = default;

const ScriptCallFrame& ScriptCallStack::at(size_t index) const
{
    ASSERT(m_frames.size() > index);
    return m_frames[index];
}

size_t ScriptCallStack::size() const
{
    return m_frames.size();
}

const ScriptCallFrame* ScriptCallStack::firstNonNativeCallFrame() const
{
    if (!m_frames.size())
        return nullptr;

    for (const auto& frame : m_frames) {
        if (!frame.isNative())
            return &frame;
    }

    return nullptr;
}

void ScriptCallStack::append(const ScriptCallFrame& frame)
{
    m_frames.append(frame);
}

void ScriptCallStack::removeParentStackTrace()
{
    m_parentStackTrace = nullptr;
}

bool ScriptCallStack::isEqual(ScriptCallStack* o) const
{
    if (!o)
        return false;

    size_t frameCount = o->m_frames.size();
    if (frameCount != m_frames.size())
        return false;

    for (size_t i = 0; i < frameCount; ++i) {
        if (!m_frames[i].isEqual(o->m_frames[i]))
            return false;
    }

    return true;
}

Ref<JSON::ArrayOf<Protocol::Console::CallFrame>> ScriptCallStack::buildInspectorArray() const
{
    auto frames = JSON::ArrayOf<Protocol::Console::CallFrame>::create();
    for (size_t i = 0; i < m_frames.size(); i++)
        frames->addItem(m_frames.at(i).buildInspectorObject());
    return frames;
}

Ref<Protocol::Console::StackTrace> ScriptCallStack::buildInspectorObject() const
{
    auto frames = JSON::ArrayOf<Protocol::Console::CallFrame>::create();
    for (const auto& item : m_frames)
        frames->addItem(item.buildInspectorObject());

    auto stackTrace = Protocol::Console::StackTrace::create()
        .setCallFrames(WTFMove(frames))
        .release();

    if (m_truncated)
        stackTrace->setTruncated(true);

    if (m_parentStackTrace)
        stackTrace->setParentStackTrace(m_parentStackTrace->buildInspectorObject());

    return stackTrace;
}

} // namespace Inspector
