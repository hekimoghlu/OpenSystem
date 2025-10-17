/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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
#include <wtf/AutomaticThread.h>

#include <wtf/DataLog.h>
#include <wtf/Threading.h>

namespace WTF {

static constexpr bool verbose = false;

Ref<AutomaticThreadCondition> AutomaticThreadCondition::create()
{
    return adoptRef(*new AutomaticThreadCondition);
}

AutomaticThreadCondition::AutomaticThreadCondition() = default;

AutomaticThreadCondition::~AutomaticThreadCondition() = default;

void AutomaticThreadCondition::notifyOne(const AbstractLocker& locker)
{
    for (AutomaticThread* thread : m_threads) {
        if (thread->isWaiting(locker)) {
            thread->notify(locker);
            return;
        }
    }

    for (AutomaticThread* thread : m_threads) {
        if (!thread->hasUnderlyingThread(locker)) {
            thread->start(locker);
            return;
        }
    }

    m_condition.notifyOne();
}

void AutomaticThreadCondition::notifyAll(const AbstractLocker& locker)
{
    m_condition.notifyAll();

    for (AutomaticThread* thread : m_threads) {
        if (thread->isWaiting(locker))
            thread->notify(locker);
        else if (!thread->hasUnderlyingThread(locker))
            thread->start(locker);
    }
}

void AutomaticThreadCondition::wait(Lock& lock)
{
    m_condition.wait(lock);
}

bool AutomaticThreadCondition::waitFor(Lock& lock, Seconds time)
{
    return m_condition.waitFor(lock, time);
}

void AutomaticThreadCondition::add(const AbstractLocker&, AutomaticThread* thread)
{
    ASSERT(!m_threads.contains(thread));
    m_threads.append(thread);
}

void AutomaticThreadCondition::remove(const AbstractLocker&, AutomaticThread* thread)
{
    m_threads.removeFirst(thread);
    ASSERT(!m_threads.contains(thread));
}

bool AutomaticThreadCondition::contains(const AbstractLocker&, AutomaticThread* thread)
{
    return m_threads.contains(thread);
}

AutomaticThread::AutomaticThread(const AbstractLocker& locker, Box<Lock> lock, Ref<AutomaticThreadCondition>&& condition, Seconds timeout)
    : AutomaticThread(locker, lock, WTFMove(condition), ThreadType::Unknown, timeout)
{
}

AutomaticThread::AutomaticThread(const AbstractLocker& locker, Box<Lock> lock, Ref<AutomaticThreadCondition>&& condition, ThreadType type, Seconds timeout)
    : m_lock(lock)
    , m_condition(WTFMove(condition))
    , m_timeout(timeout)
    , m_threadType(type)
{
    if (verbose)
        dataLog(RawPointer(this), ": Allocated AutomaticThread.\n");
    m_condition->add(locker, this);
}

AutomaticThread::~AutomaticThread()
{
    if (verbose)
        dataLog(RawPointer(this), ": Deleting AutomaticThread.\n");
    Locker locker { *m_lock };
    
    // It's possible that we're in a waiting state with the thread shut down. This is a goofy way to
    // die, but it could happen.
    m_condition->remove(locker, this);
}

bool AutomaticThread::tryStop(const AbstractLocker&)
{
    if (!m_isRunning)
        return true;
    if (m_hasUnderlyingThread)
        return false;
    m_isRunning = false;
    return true;
}

bool AutomaticThread::isWaiting(const AbstractLocker& locker)
{
    return hasUnderlyingThread(locker) && m_isWaiting;
}

bool AutomaticThread::notify(const AbstractLocker& locker)
{
    ASSERT_UNUSED(locker, hasUnderlyingThread(locker));
    m_isWaiting = false;
    return m_waitCondition.notifyOne();
}

void AutomaticThread::join()
{
    Locker locker { *m_lock };
    while (m_isRunning)
        m_isRunningCondition.wait(*m_lock);
}

void AutomaticThread::start(const AbstractLocker&)
{
    RELEASE_ASSERT(m_isRunning);
    
    RefPtr<AutomaticThread> preserveThisForThread = this;
    
    m_hasUnderlyingThread = true;
    
    Thread::create(
        name(),
        [=, this] () {
            if (verbose)
                dataLog(RawPointer(this), ": Running automatic thread!\n");
            
            RefPtr<AutomaticThread> thread = preserveThisForThread;
            thread->threadDidStart();
            
            if (ASSERT_ENABLED) {
                Locker locker { *m_lock };
                ASSERT(m_condition->contains(locker, this));
            }
            
            auto stopImpl = [&] (const AbstractLocker& locker) {
                thread->threadIsStopping(locker);
                thread->m_hasUnderlyingThread = false;
            };
            
            auto stopPermanently = [&] (const AbstractLocker& locker) {
                m_isRunning = false;
                m_isRunningCondition.notifyAll();
                stopImpl(locker);
            };
            
            auto stopForTimeout = [&] (const AbstractLocker& locker) {
                stopImpl(locker);
            };
            
            for (;;) {
                {
                    Locker locker { *m_lock };
                    for (;;) {
                        PollResult result = poll(locker);
                        if (result == PollResult::Work)
                            break;
                        if (result == PollResult::Stop)
                            return stopPermanently(locker);
                        RELEASE_ASSERT(result == PollResult::Wait);

                        // Shut the thread down after a timeout.
                        m_isWaiting = true;
                        bool awokenByNotify =
                            m_waitCondition.waitFor(*m_lock, m_timeout);
                        if (verbose && !awokenByNotify && !m_isWaiting)
                            dataLog(RawPointer(this), ": waitFor timed out, but notified via m_isWaiting flag!\n");
                        if (m_isWaiting && shouldSleep(locker)) {
                            m_isWaiting = false;
                            if (verbose)
                                dataLog(RawPointer(this), ": Going to sleep!\n");
                            // It's important that we don't release the lock until we have completely
                            // indicated that the thread is kaput. Otherwise we'll have a a notify
                            // race that manifests as a deadlock on VM shutdown.
                            return stopForTimeout(locker);
                        }
                    }
                }
                
                WorkResult result = work();
                if (result == WorkResult::Stop) {
                    Locker locker { *m_lock };
                    return stopPermanently(locker);
                }
                RELEASE_ASSERT(result == WorkResult::Continue);
            }
        }, m_threadType)->detach();
}

void AutomaticThread::threadDidStart()
{
}

void AutomaticThread::threadIsStopping(const AbstractLocker&)
{
}

} // namespace WTF

