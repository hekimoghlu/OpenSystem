/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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

#include "InspectorProtocolObjects.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>

namespace Inspector {

class ScriptCallFrame;
class ScriptCallStack;

class AsyncStackTrace : public RefCounted<AsyncStackTrace> {
public:
    enum class State : uint8_t {
        Pending,
        Active,
        Dispatched,
        Canceled,
    };

    static Ref<AsyncStackTrace> create(Ref<ScriptCallStack>&&, bool singleShot, RefPtr<AsyncStackTrace> parent);

    bool isPending() const;
    bool isLocked() const;

    JS_EXPORT_PRIVATE const ScriptCallFrame& at(size_t) const;
    JS_EXPORT_PRIVATE size_t size() const;
    JS_EXPORT_PRIVATE bool topCallFrameIsBoundary() const;
    bool truncated() const { return m_truncated; }

    const RefPtr<AsyncStackTrace>& parentStackTrace() const { return m_parent; }

    void willDispatchAsyncCall(size_t maxDepth);
    void didDispatchAsyncCall();
    void didCancelAsyncCall();

    Ref<Protocol::Console::StackTrace> buildInspectorObject() const;

    JS_EXPORT_PRIVATE ~AsyncStackTrace();

private:
    AsyncStackTrace(Ref<ScriptCallStack>&&, bool, RefPtr<AsyncStackTrace>);

    void truncate(size_t maxDepth);
    void remove();

    Ref<ScriptCallStack> m_callStack;
    RefPtr<AsyncStackTrace> m_parent;
    unsigned m_childCount { 0 };
    State m_state { State::Pending };
    bool m_truncated { false };
    bool m_singleShot { true };
};

} // namespace Inspector
