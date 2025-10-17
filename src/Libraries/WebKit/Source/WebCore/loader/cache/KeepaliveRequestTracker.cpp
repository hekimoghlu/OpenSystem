/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#include "KeepaliveRequestTracker.h"

namespace WebCore {

const uint64_t maxInflightKeepaliveBytes { 65536 }; // 64 kibibytes as per Fetch specification.

KeepaliveRequestTracker::~KeepaliveRequestTracker()
{
    auto inflightRequests = WTFMove(m_inflightKeepaliveRequests);
    for (auto& resource : inflightRequests)
        resource->removeClient(*this);
}

bool KeepaliveRequestTracker::tryRegisterRequest(CachedResource& resource)
{
    ASSERT(resource.options().keepAlive);
    auto body = resource.resourceRequest().httpBody();
    if (!body)
        return true;

    uint64_t bodySize = body->lengthInBytes();
    if (m_inflightKeepaliveBytes + bodySize > maxInflightKeepaliveBytes)
        return false;

    registerRequest(resource);
    return true;
}

void KeepaliveRequestTracker::registerRequest(CachedResource& resource)
{
    ASSERT(resource.options().keepAlive);
    RefPtr body = resource.resourceRequest().httpBody();
    if (!body)
        return;
    ASSERT(!m_inflightKeepaliveRequests.contains(&resource));
    m_inflightKeepaliveRequests.append(&resource);
    m_inflightKeepaliveBytes += body->lengthInBytes();
    ASSERT(m_inflightKeepaliveBytes <= maxInflightKeepaliveBytes);

    resource.addClient(*this);
}

void KeepaliveRequestTracker::notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    unregisterRequest(resource);
}

void KeepaliveRequestTracker::unregisterRequest(CachedResource& resource)
{
    ASSERT(resource.options().keepAlive);

    m_inflightKeepaliveBytes -= resource.resourceRequest().httpBody()->lengthInBytes();
    ASSERT(m_inflightKeepaliveBytes <= maxInflightKeepaliveBytes);

    resource.removeClient(*this);
    bool wasRemoved = m_inflightKeepaliveRequests.removeFirst(&resource); // May destroy |resource|.
    ASSERT_UNUSED(wasRemoved, wasRemoved);
}

}
