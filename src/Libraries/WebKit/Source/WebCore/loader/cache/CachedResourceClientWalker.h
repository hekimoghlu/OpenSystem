/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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

#include "CachedResource.h"
#include "CachedResourceClient.h"
#include "CachedResourceHandle.h"
#include <wtf/FixedVector.h>
#include <wtf/HashCountedSet.h>

namespace WebCore {

// Call this "walker" instead of iterator so people won't expect Qt or STL-style iterator interface.
// Just keep calling next() on this. It's safe from deletions of items.
template<typename T>
class CachedResourceClientWalker {
public:
    CachedResourceClientWalker(const CachedResource& resource)
        : m_resource(const_cast<CachedResource*>(&resource))
        , m_clientVector(resource.m_clients.computeSize())
    {
        size_t clientIndex = 0;
        for (auto client : resource.m_clients)
            m_clientVector[clientIndex++] = client.key;
    }

    T* next()
    {
        size_t size = m_clientVector.size();
        while (m_index < size) {
            auto& next = m_clientVector[m_index++];
            if (next && m_resource->m_clients.contains(*next)) {
                RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(T::expectedType() == CachedResourceClient::expectedType() || next->resourceClientType() == T::expectedType());
                return static_cast<T*>(next.get());
            }
        }
        return nullptr;
    }

private:
    CachedResourceHandle<CachedResource> m_resource;
    FixedVector<SingleThreadWeakPtr<CachedResourceClient>> m_clientVector;
    size_t m_index { 0 };
};

} // namespace WebCore
