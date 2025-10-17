/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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

#if ENABLE(THREADING_LIBDISPATCH)

#include <dispatch/dispatch.h>

namespace WTF {

class ParallelEnvironment {
    WTF_MAKE_FAST_ALLOCATED;
public:
    typedef void (*ThreadFunction)(void*);

    ParallelEnvironment(ThreadFunction threadFunction, size_t sizeOfParameter, int requestedJobNumber)
        : m_threadFunction(threadFunction)
        , m_sizeOfParameter(sizeOfParameter)
        , m_numberOfJobs(requestedJobNumber)
    {
        // We go with the requested number of jobs. libdispatch will distribute the work optimally.
        ASSERT_ARG(requestedJobNumber, requestedJobNumber > 0);
    }

    int numberOfJobs()
    {
        return m_numberOfJobs;
    }

    void execute(unsigned char* parameters)
    {
        static dispatch_queue_t globalQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        dispatch_apply(m_numberOfJobs, globalQueue, ^(size_t i) { (*m_threadFunction)(parameters + (m_sizeOfParameter * i)); });
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }

private:
    ThreadFunction m_threadFunction;
    size_t m_sizeOfParameter;
    int m_numberOfJobs;
};

} // namespace WTF

#endif // ENABLE(THREADING_LIBDISPATCH)
