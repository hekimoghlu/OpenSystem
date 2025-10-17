/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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

#include "ScriptExecutionContext.h"
#include <wtf/Lock.h>

namespace WebCore {

// A helper class to safely dereference the callback objects held by
// SQLStatement and SQLTransaction on the proper thread. The 'wrapped'
// callback is dereferenced:
// - by destructing the enclosing wrapper - on any thread
// - by calling clear() - on any thread
// - by unwrapping and then dereferencing normally - on context thread only
template<typename T> class SQLCallbackWrapper {
public:
    SQLCallbackWrapper(RefPtr<T>&& callback, ScriptExecutionContext* scriptExecutionContext)
        : m_callback(WTFMove(callback))
        , m_scriptExecutionContext(m_callback ? scriptExecutionContext : 0)
    {
        ASSERT(!m_callback || (m_scriptExecutionContext.get() && m_scriptExecutionContext->isContextThread()));
    }

    ~SQLCallbackWrapper()
    {
        clear();
    }

    void clear()
    {
        ScriptExecutionContext* scriptExecutionContextPtr;
        T* callback;
        {
            Locker locker { m_lock };
            if (!m_callback) {
                ASSERT(!m_scriptExecutionContext);
                return;
            }
            if (m_scriptExecutionContext->isContextThread()) {
                m_callback = nullptr;
                m_scriptExecutionContext = nullptr;
                return;
            }
            scriptExecutionContextPtr = m_scriptExecutionContext.leakRef();
            callback = m_callback.leakRef();
        }
        scriptExecutionContextPtr->postTask({
            ScriptExecutionContext::Task::CleanupTask,
            [callback, scriptExecutionContextPtr] (ScriptExecutionContext& context) {
                ASSERT_UNUSED(context, &context == scriptExecutionContextPtr && context.isContextThread());
                callback->deref();
                scriptExecutionContextPtr->deref();
            }
        });
    }

    RefPtr<T> unwrap()
    {
        Locker locker { m_lock };
        ASSERT(!m_callback || m_scriptExecutionContext->isContextThread());
        m_scriptExecutionContext = nullptr;
        return WTFMove(m_callback);
    }

    // Useful for optimizations only, please test the return value of unwrap to be sure.
    // FIXME: This is not thread-safe.
    bool hasCallback() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { return m_callback; }

private:
    Lock m_lock;
    RefPtr<T> m_callback WTF_GUARDED_BY_LOCK(m_lock);
    RefPtr<ScriptExecutionContext> m_scriptExecutionContext WTF_GUARDED_BY_LOCK(m_lock);
};

} // namespace WebCore
