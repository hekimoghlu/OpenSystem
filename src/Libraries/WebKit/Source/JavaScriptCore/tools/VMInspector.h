/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#include "CallFrame.h"
#include "VM.h"
#include <wtf/DoublyLinkedList.h>
#include <wtf/Expected.h>
#include <wtf/IterationStatus.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class VMInspector {
    WTF_MAKE_TZONE_ALLOCATED(VMInspector);
    WTF_MAKE_NONCOPYABLE(VMInspector);
    VMInspector() = default;
public:
    enum class Error {
        None,
        TimedOut
    };

    static VMInspector& singleton();

    void add(VM*);
    void remove(VM*);
    ALWAYS_INLINE static bool isValidVM(VM* vm)
    {
        return vm == m_recentVM ? true : isValidVMSlow(vm);
    }

    Lock& getLock() WTF_RETURNS_LOCK(m_lock) { return m_lock; }

    void iterate(const Invocable<IterationStatus(VM&)> auto& functor) WTF_REQUIRES_LOCK(m_lock)
    {
        for (VM* vm = m_vmList.head(); vm; vm = vm->next()) {
            IterationStatus status = functor(*vm);
            if (status == IterationStatus::Done)
                return;
        }
    }

    JS_EXPORT_PRIVATE static void forEachVM(Function<IterationStatus(VM&)>&&);
    JS_EXPORT_PRIVATE static void dumpVMs();

    // Returns null if the callFrame doesn't actually correspond to any active VM.
    JS_EXPORT_PRIVATE static VM* vmForCallFrame(CallFrame*);

    Expected<bool, Error> isValidExecutableMemory(void*) WTF_REQUIRES_LOCK(m_lock);
    Expected<CodeBlock*, Error> codeBlockForMachinePC(void*) WTF_REQUIRES_LOCK(m_lock);

    JS_EXPORT_PRIVATE static bool currentThreadOwnsJSLock(VM*);
    JS_EXPORT_PRIVATE static void gc(VM*);
    JS_EXPORT_PRIVATE static void edenGC(VM*);
    JS_EXPORT_PRIVATE static bool isInHeap(Heap*, void*);
    JS_EXPORT_PRIVATE static bool isValidCell(Heap*, JSCell*);
    JS_EXPORT_PRIVATE static bool isValidCodeBlock(VM*, CodeBlock*);
    JS_EXPORT_PRIVATE static CodeBlock* codeBlockForFrame(VM*, CallFrame* topCallFrame, unsigned frameNumber);
    JS_EXPORT_PRIVATE static void dumpCallFrame(VM*, CallFrame*, unsigned framesToSkip = 0);
    JS_EXPORT_PRIVATE static void dumpRegisters(CallFrame*);
    JS_EXPORT_PRIVATE static void dumpStack(VM*, CallFrame* topCallFrame, unsigned framesToSkip = 0);
    JS_EXPORT_PRIVATE static void dumpValue(JSValue);
    JS_EXPORT_PRIVATE static void dumpCellMemory(JSCell*);
    JS_EXPORT_PRIVATE static void dumpCellMemoryToStream(JSCell*, PrintStream&);
    JS_EXPORT_PRIVATE static void dumpSubspaceHashes(VM*);

#if USE(JSVALUE64)
    static bool verifyCell(VM&, JSCell*);
#endif

private:
    JS_EXPORT_PRIVATE static bool isValidVMSlow(VM*);

    Lock m_lock;
    DoublyLinkedList<VM> m_vmList WTF_GUARDED_BY_LOCK(m_lock);
    JS_EXPORT_PRIVATE static VM* m_recentVM;
};

} // namespace JSC
