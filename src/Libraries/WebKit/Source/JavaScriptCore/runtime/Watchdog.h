/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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

#include <wtf/Lock.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace JSC {

class CallFrame;
class JSGlobalObject;
class VM;

class Watchdog : public ThreadSafeRefCounted<Watchdog> {
    WTF_MAKE_TZONE_ALLOCATED(Watchdog);
public:
    class Scope;

    Watchdog(VM*);
    void willDestroyVM(VM*);

    typedef bool (*ShouldTerminateCallback)(JSGlobalObject*, void* data1, void* data2);
    void setTimeLimit(Seconds limit, ShouldTerminateCallback = nullptr, void* data1 = nullptr, void* data2 = nullptr);

    bool shouldTerminate(JSGlobalObject*);

    bool isActive() const { return m_hasEnteredVM; }

    bool hasTimeLimit();
    void enteredVM();
    void exitedVM();

    static constexpr Seconds noTimeLimit = Seconds::infinity();

private:
    void startTimer(Seconds timeLimit);
    void stopTimer();

    bool m_hasEnteredVM { false };

    Lock m_lock; // Guards access to m_vm.
    VM* m_vm { nullptr };

    Seconds m_timeLimit { noTimeLimit };
    Seconds m_cpuDeadline { noTimeLimit };
    MonotonicTime m_deadline { MonotonicTime::infinity() };

    ShouldTerminateCallback m_callback { nullptr };
    void* m_callbackData1 { nullptr };
    void* m_callbackData2 { nullptr };
};

} // namespace JSC
