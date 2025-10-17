/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#ifndef BlockingResponseMap_h
#define BlockingResponseMap_h

#include <wtf/Condition.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>

template<typename T>
class BlockingResponseMap {
WTF_MAKE_NONCOPYABLE(BlockingResponseMap);
public:
    BlockingResponseMap() : m_canceled(false) { }
    ~BlockingResponseMap() { ASSERT(m_responses.isEmpty()); }

    std::unique_ptr<T> waitForResponse(uint64_t requestID)
    {
        while (true) {
            Locker locker { m_responsesLock };

            if (m_canceled)
                return nullptr;

            if (std::unique_ptr<T> response = m_responses.take(requestID))
                return response;

            m_condition.wait(m_responsesLock);
        }

        return nullptr;
    }

    void didReceiveResponse(uint64_t requestID, std::unique_ptr<T> response)
    {
        Locker locker { m_responsesLock };
        ASSERT(!m_responses.contains(requestID));

        m_responses.set(requestID, WTFMove(response));

        // FIXME: Could get a slight speed-up from using notifyOne().
        m_condition.notifyAll();
    }

    void cancel()
    {
        m_canceled = true;

        // FIXME: Could get a slight speed-up from using notifyOne().
        m_condition.notifyAll();
    }

private:
    Lock m_responsesLock;
    Condition m_condition;

    HashMap<uint64_t, std::unique_ptr<T>> m_responses WTF_GUARDED_BY_LOCK(m_responsesLock);
    bool m_canceled;
};

#endif // BlockingResponseMap_h
