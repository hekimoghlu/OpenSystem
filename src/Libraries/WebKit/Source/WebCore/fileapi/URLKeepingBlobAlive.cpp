/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#include "URLKeepingBlobAlive.h"

#include "ThreadableBlobRegistry.h"
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

URLKeepingBlobAlive::URLKeepingBlobAlive(const URL& url, const std::optional<SecurityOriginData>& topOrigin)
    : m_url(url)
    , m_topOrigin(topOrigin)
{
    registerBlobURLHandleIfNecessary();
}

URLKeepingBlobAlive::~URLKeepingBlobAlive()
{
    unregisterBlobURLHandleIfNecessary();
}

void URLKeepingBlobAlive::clear()
{
    unregisterBlobURLHandleIfNecessary();
    m_url = { };
    m_topOrigin = std::nullopt;
}

URLKeepingBlobAlive& URLKeepingBlobAlive::operator=(URLKeepingBlobAlive&& other)
{
    if (&other == this)
        return *this;

    unregisterBlobURLHandleIfNecessary();
    m_url = std::exchange(other.m_url, URL { });
    m_topOrigin = std::exchange(other.m_topOrigin, { });
    return *this;
}

void URLKeepingBlobAlive::registerBlobURLHandleIfNecessary()
{
    if (m_url.protocolIsBlob())
        ThreadableBlobRegistry::registerBlobURLHandle(m_url, m_topOrigin);
}

void URLKeepingBlobAlive::unregisterBlobURLHandleIfNecessary()
{
    if (m_url.protocolIsBlob())
        ThreadableBlobRegistry::unregisterBlobURLHandle(m_url, m_topOrigin);
}

URLKeepingBlobAlive URLKeepingBlobAlive::isolatedCopy() const
{
    return { m_url.isolatedCopy(), crossThreadCopy(m_topOrigin) };
}

} // namespace WebCore
