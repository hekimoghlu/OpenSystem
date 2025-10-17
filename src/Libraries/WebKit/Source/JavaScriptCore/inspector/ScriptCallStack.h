/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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

#include "AsyncStackTrace.h"
#include "ScriptCallFrame.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace Inspector {

class AsyncStackTrace;

class ScriptCallStack : public RefCounted<ScriptCallStack> {
public:
    static constexpr size_t maxCallStackSizeToCapture = 200;
    
    static Ref<ScriptCallStack> create();
    static Ref<ScriptCallStack> create(Vector<ScriptCallFrame>&&, bool truncated = false, AsyncStackTrace* parentStackTrace = nullptr);

    JS_EXPORT_PRIVATE ~ScriptCallStack();

    JS_EXPORT_PRIVATE const ScriptCallFrame& at(size_t) const;
    JS_EXPORT_PRIVATE size_t size() const;
    bool truncated() const { return m_truncated; }

    const RefPtr<AsyncStackTrace>& parentStackTrace() const { return m_parentStackTrace; }
    void removeParentStackTrace();

    JS_EXPORT_PRIVATE const ScriptCallFrame* firstNonNativeCallFrame() const;

    void append(const ScriptCallFrame&);

    JS_EXPORT_PRIVATE bool isEqual(ScriptCallStack*) const;

    Ref<JSON::ArrayOf<Protocol::Console::CallFrame>> buildInspectorArray() const;
    JS_EXPORT_PRIVATE Ref<Protocol::Console::StackTrace> buildInspectorObject() const;

private:
    ScriptCallStack();
    ScriptCallStack(Vector<ScriptCallFrame>&&, bool truncated = false, AsyncStackTrace* parentStackTrace = nullptr);

    Vector<ScriptCallFrame> m_frames;
    bool m_truncated { false };

    RefPtr<AsyncStackTrace> m_parentStackTrace;
};

} // namespace Inspector
