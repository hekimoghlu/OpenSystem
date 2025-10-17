/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#include "WebSQLiteDatabaseTracker.h"

#include <WebCore/SQLiteDatabaseTracker.h>

namespace WebKit {
using namespace WebCore;

Ref<WebSQLiteDatabaseTracker> WebSQLiteDatabaseTracker::create(IsHoldingLockedFilesHandler&& isHoldingLockedFilesHandler)
{
    return adoptRef(*new WebSQLiteDatabaseTracker(WTFMove(isHoldingLockedFilesHandler)));
}

WebSQLiteDatabaseTracker::WebSQLiteDatabaseTracker(IsHoldingLockedFilesHandler&& isHoldingLockedFilesHandler)
    : m_isHoldingLockedFilesHandler(WTFMove(isHoldingLockedFilesHandler))
{
    ASSERT(RunLoop::isMain());
    SQLiteDatabaseTracker::setClient(this);
}

WebSQLiteDatabaseTracker::~WebSQLiteDatabaseTracker()
{
    ASSERT(RunLoop::isMain());
    SQLiteDatabaseTracker::setClient(nullptr);

    if (m_currentHystererisID)
        setIsHoldingLockedFiles(false);
}

void WebSQLiteDatabaseTracker::setIsSuspended(bool isSuspended)
{
    ASSERT(RunLoop::isMain());

    Locker locker { m_lock };
    m_isSuspended = isSuspended;
}

void WebSQLiteDatabaseTracker::willBeginFirstTransaction()
{
    Locker locker { m_lock };
    if (m_currentHystererisID) {
        // Cancel previous hysteresis task.
        ++m_currentHystererisID;
        return;
    }

    RunLoop::protectedMain()->dispatch([weakThis = WeakPtr { *this }] {
        if (RefPtr protectedThis = weakThis.get()) {
            Locker locker { protectedThis->m_lock };
            protectedThis->setIsHoldingLockedFiles(true);
        }
    });
}

void WebSQLiteDatabaseTracker::didFinishLastTransaction()
{
    Locker locker { m_lock };
    RunLoop::protectedMain()->dispatchAfter(1_s, [weakThis = WeakPtr { *this }, hystererisID = ++m_currentHystererisID] {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return;

        Locker locker { protectedThis->m_lock };
        if (protectedThis->m_currentHystererisID != hystererisID)
            return; // Cancelled.

        protectedThis->m_currentHystererisID = 0;
        protectedThis->setIsHoldingLockedFiles(false);
    });
}

void WebSQLiteDatabaseTracker::setIsHoldingLockedFiles(bool isHoldingLockedFiles)
{
    if (m_isSuspended)
        return;

    m_isHoldingLockedFilesHandler(isHoldingLockedFiles);
}

} // namespace WebKit
