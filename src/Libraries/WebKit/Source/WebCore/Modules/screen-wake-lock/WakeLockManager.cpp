/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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
#include "WakeLockManager.h"

#include "Document.h"
#include "SleepDisabler.h"
#include "VisibilityState.h"
#include "WakeLockSentinel.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WakeLockManager);

WakeLockManager::WakeLockManager(Document& document)
    : m_document(document)
{
    m_document.registerForVisibilityStateChangedCallbacks(*this);
}

WakeLockManager::~WakeLockManager()
{
    m_document.unregisterForVisibilityStateChangedCallbacks(*this);
}

void WakeLockManager::ref() const
{
    m_document.ref();
}

void WakeLockManager::deref() const
{
    m_document.deref();
}

void WakeLockManager::addWakeLock(Ref<WakeLockSentinel>&& lock, std::optional<PageIdentifier> pageID)
{
    auto type = lock->type();
    auto& locks = m_wakeLocks.ensure(type, [] { return Vector<RefPtr<WakeLockSentinel>>(); }).iterator->value;
    ASSERT(!locks.contains(lock.ptr()));
    locks.append(WTFMove(lock));

    if (locks.size() != 1)
        return;

    switch (type) {
    case WakeLockType::Screen:
        m_screenLockDisabler = makeUnique<SleepDisabler>("Screen Wake Lock"_s, PAL::SleepDisabler::Type::Display, pageID);
        break;
    }
}

void WakeLockManager::removeWakeLock(WakeLockSentinel& lock)
{
    auto it = m_wakeLocks.find(lock.type());
    if (it == m_wakeLocks.end())
        return;
    auto& locks = it->value;
    locks.removeFirst(&lock);
    ASSERT(!locks.contains(&lock));

    if (!locks.isEmpty())
        return;

    m_wakeLocks.remove(it);
    switch (lock.type()) {
    case WakeLockType::Screen:
        m_screenLockDisabler = nullptr;
        break;
    }
}

// https://www.w3.org/TR/screen-wake-lock/#handling-document-loss-of-visibility
void WakeLockManager::visibilityStateChanged()
{
    if (m_document.visibilityState() != VisibilityState::Hidden)
        return;

    releaseAllLocks(WakeLockType::Screen);
}

void WakeLockManager::releaseAllLocks(WakeLockType type)
{
    auto it = m_wakeLocks.find(type);
    if (it == m_wakeLocks.end())
        return;

    auto& locks = it->value;
    while (!locks.isEmpty()) {
        RefPtr lock = *locks.begin();
        lock->release(*this);
    }
}

} // namespace WebCore
