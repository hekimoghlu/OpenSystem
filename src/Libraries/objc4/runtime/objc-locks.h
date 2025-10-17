/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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
/***********************************************************************
* objc-locks.h
* Declarations of all locks used in the runtime.
**********************************************************************/

#ifndef _OBJC_LOCKS_H
#define _OBJC_LOCKS_H

#include "objc-config.h"
#include "InitWrappers.h"

// fork() safety requires careful tracking of all locks used in the runtime.
// Thou shalt not declare any locks outside this file.

// Lock ordering is declared in _objc_fork_prepare()
// and is enforced by lockdebug.

// ExplicitInit wrapper around a lock. Convertible to the underlying lock type
// and forwards basic lock/unlock.
template <typename Lock>
class ExplicitInitLock: public objc::ExplicitInit<Lock> {
public:
    operator Lock &() {
        return this->get();
    }

    void lock() {
        this->get().lock();
    }

    void unlock() {
        this->get().unlock();
    }

    void reset() {
        this->get().reset();
    }

    bool tryLock() {
        return this->get().tryLock();
    }
};


extern ExplicitInitLock<mutex_t> classInitLock;
extern ExplicitInitLock<mutex_t> pendingInitializeMapLock;
extern ExplicitInitLock<mutex_t> selLock;
#if CONFIG_USE_CACHE_LOCK
extern ExplicitInitLock<mutex_t> cacheUpdateLock;
#endif
extern ExplicitInitLock<recursive_mutex_t> loadMethodLock;
extern ExplicitInitLock<mutex_t> crashlog_lock;
extern ExplicitInitLock<spinlock_t> objcMsgLogLock;
extern ExplicitInitLock<mutex_t> AltHandlerDebugLock;
extern ExplicitInitLock<mutex_t> AssociationsManagerLock;
extern objc::ExplicitInit<StripedMap<spinlock_t>> PropertyLocks;
extern objc::ExplicitInit<StripedMap<spinlock_t>> StructLocks;
extern objc::ExplicitInit<StripedMap<spinlock_t>> CppObjectLocks;
extern ExplicitInitLock<mutex_t> runtimeLock;
extern ExplicitInitLock<mutex_t> DemangleCacheLock;

// SideTable lock is buried awkwardly. Call a function to manipulate it.
extern void SideTableLockAll();
extern void SideTableUnlockAll();
extern void SideTableForceResetAll();
extern spinlock_t *SideTableGetLock(unsigned n);

#endif
