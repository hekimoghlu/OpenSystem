/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "FetchBodySource.h"

#include "FetchResponse.h"

namespace WebCore {

FetchBodySource::FetchBodySource(FetchBodyOwner& bodyOwner)
    : m_bodyOwner(bodyOwner)
{
}

FetchBodySource::~FetchBodySource() = default;

void FetchBodySource::setActive()
{
    ASSERT(m_bodyOwner);
    ASSERT(!m_pendingActivity);
    if (m_bodyOwner)
        m_pendingActivity = m_bodyOwner->makePendingActivity(*m_bodyOwner);
}

void FetchBodySource::setInactive()
{
    ASSERT(m_bodyOwner);
    ASSERT(m_pendingActivity);
    m_pendingActivity = nullptr;
}

void FetchBodySource::doStart()
{
    ASSERT(m_bodyOwner);
    if (m_bodyOwner)
        m_bodyOwner->consumeBodyAsStream();
}

void FetchBodySource::doPull()
{
    ASSERT(m_bodyOwner);
    if (m_bodyOwner)
        m_bodyOwner->feedStream();
}

void FetchBodySource::doCancel()
{
    m_isCancelling = true;
    if (!m_bodyOwner)
        return;

    m_bodyOwner->cancel();
    m_bodyOwner = nullptr;
}

void FetchBodySource::close()
{
#if ASSERT_ENABLED
    ASSERT(!m_isClosed);
    m_isClosed = true;
#endif

    controller().close();
    clean();
    m_bodyOwner = nullptr;
}

void FetchBodySource::error(const Exception& value)
{
    controller().error(value);
    clean();
    m_bodyOwner = nullptr;
}

} // namespace WebCore
