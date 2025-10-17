/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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

#include <atomic>
#include <utility>

namespace WebKit {

class ScopedRenderingResourcesRequest {
public:
    ScopedRenderingResourcesRequest() = default;
    ScopedRenderingResourcesRequest(ScopedRenderingResourcesRequest&& other)
        : m_requested(std::exchange(other.m_requested, false))
    {
    }
    ~ScopedRenderingResourcesRequest()
    {
        reset();
    }
    ScopedRenderingResourcesRequest& operator=(ScopedRenderingResourcesRequest&& other)
    {
        if (this != &other) {
            reset();
            m_requested = std::exchange(other.m_requested, false);
        }
        return *this;
    }
    bool isRequested() const { return m_requested; }
    void reset()
    {
        if (!m_requested)
            return;
        ASSERT(s_requests);
        --s_requests;
        if (!s_requests)
            scheduleFreeRenderingResources();
    }
    static ScopedRenderingResourcesRequest acquire()
    {
        return { DidRequest };
    }
private:
    static void scheduleFreeRenderingResources();
    static void freeRenderingResources();
    enum RequestState { DidRequest };
    ScopedRenderingResourcesRequest(RequestState)
        : m_requested(true)
    {
        ++s_requests;
    }
    static std::atomic<unsigned> s_requests;
    bool m_requested { false };
    friend bool hasOutstandingRenderingResourceUsage();
};

inline bool hasOutstandingRenderingResourceUsage()
{
    return ScopedRenderingResourcesRequest::s_requests;
}

}
