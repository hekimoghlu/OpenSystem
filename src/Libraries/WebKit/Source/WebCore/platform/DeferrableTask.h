/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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

#include <wtf/Function.h>
#include <wtf/MainThread.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class MainThreadDeferrableTask;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MainThreadDeferrableTask> : std::true_type { };
}

namespace WebCore {

class MainThreadDeferrableTask : public CanMakeWeakPtr<MainThreadDeferrableTask> {
public:
    MainThreadDeferrableTask() = default;

    void close()
    {
        cancelTask();
        m_isClosed = true;
    }

    void cancelTask()
    {
        weakPtrFactory().revokeAll();
        m_isPending = false;
    }

    bool isPending() const { return m_isPending; }

    void scheduleTask(Function<void()>&& task)
    {
        if (m_isClosed)
            return;

        cancelTask();

        m_isPending = true;
        callOnMainThread([weakThis = WeakPtr { *this }, task = WTFMove(task)] {
            if (!weakThis)
                return;
            ASSERT(weakThis->isPending());
            weakThis->m_isPending = false;
            task();
        });
    }

private:
    bool m_isPending { false };
    bool m_isClosed { false };
};

} // namespace WebCore
