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
#pragma once

#include <wtf/Condition.h>
#include <wtf/Lock.h>

namespace WTF {

// This is a traditional read-write lock implementation that enables concurrency between readers so long as
// the read critical section is long. Concurrent readers will experience contention on read().lock() and
// read().unlock() if the work inside the critical section is short. The more cores participate in reading,
// the longer the read critical section has to be for this locking scheme to be profitable.

class ReadWriteLock {
    WTF_MAKE_NONCOPYABLE(ReadWriteLock);
    WTF_MAKE_FAST_ALLOCATED;
public:
    ReadWriteLock() = default;

    // It's easiest to read lock like this:
    // 
    //     Locker locker { rwLock.read() };
    //
    // It's easiest to write lock like this:
    // 
    //     Locker locker { rwLock.write() };
    //
    WTF_EXPORT_PRIVATE void readLock();
    WTF_EXPORT_PRIVATE void readUnlock();
    WTF_EXPORT_PRIVATE void writeLock();
    WTF_EXPORT_PRIVATE void writeUnlock();
    
    class ReadLock;
    class WriteLock;

    ReadLock& read();
    WriteLock& write();

private:
    Lock m_lock;
    Condition m_cond;
    bool m_isWriteLocked WTF_GUARDED_BY_LOCK(m_lock) { false };
    unsigned m_numReaders WTF_GUARDED_BY_LOCK(m_lock) { 0 };
    unsigned m_numWaitingWriters WTF_GUARDED_BY_LOCK(m_lock) { 0 };
};

class ReadWriteLock::ReadLock : public ReadWriteLock {
public:
    bool tryLock() { return false; }
    void lock() { readLock(); }
    void unlock() { readUnlock(); }
};

class ReadWriteLock::WriteLock : public ReadWriteLock {
public:
    bool tryLock() { return false; }
    void lock() { writeLock(); }
    void unlock() { writeUnlock(); }
};
    
inline ReadWriteLock::ReadLock& ReadWriteLock::read() { return *static_cast<ReadLock*>(this); }
inline ReadWriteLock::WriteLock& ReadWriteLock::write() { return *static_cast<WriteLock*>(this); }

} // namespace WTF

using WTF::ReadWriteLock;
