/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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
#include <wtf/ReadWriteLock.h>

#include <wtf/Locker.h>

namespace WTF {

void ReadWriteLock::readLock()
{
    Locker locker { m_lock };
    while (m_isWriteLocked || m_numWaitingWriters)
        m_cond.wait(m_lock);
    m_numReaders++;
}

void ReadWriteLock::readUnlock()
{
    Locker locker { m_lock };
    m_numReaders--;
    if (!m_numReaders)
        m_cond.notifyAll();
}

void ReadWriteLock::writeLock()
{
    Locker locker { m_lock };
    while (m_isWriteLocked || m_numReaders) {
        m_numWaitingWriters++;
        m_cond.wait(m_lock);
        m_numWaitingWriters--;
    }
    m_isWriteLocked = true;
}

void ReadWriteLock::writeUnlock()
{
    Locker locker { m_lock };
    m_isWriteLocked = false;
    m_cond.notifyAll();
}

} // namespace WTF

