/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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

#if ENABLE(WEBASSEMBLY)

#include "CompilationResult.h"
#include "WasmJS.h"
#include "WasmModuleInformation.h"
#include "WasmOMGIRGenerator.h"
#include <wtf/Bag.h>
#include <wtf/CrossThreadCopier.h>
#include <wtf/SharedTask.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>

namespace JSC {

class CallLinkInfo;
class VM;

namespace Wasm {

class CalleeGroup;

class Plan : public ThreadSafeRefCounted<Plan> {
public:
    typedef void CallbackType(Plan&);
    using CompletionTask = RefPtr<SharedTask<CallbackType>>;

    static CompletionTask dontFinalize() { return createSharedTask<CallbackType>([](Plan&) { }); }
    Plan(VM&, Ref<ModuleInformation>, CompletionTask&&);

    // Note: This constructor should only be used if you are not actually building a module e.g. validation/function tests
    JS_EXPORT_PRIVATE Plan(VM&, CompletionTask&&);
    virtual JS_EXPORT_PRIVATE ~Plan();

    // If you guarantee the ordering here, you can rely on FIFO of the completion tasks being called.
    // Return false if the task plan is already completed.
    bool addCompletionTaskIfNecessary(VM&, CompletionTask&&);

    void setMode(MemoryMode mode) { m_mode = mode; }
    ALWAYS_INLINE MemoryMode mode() const { return m_mode; }

    String errorMessage() const { return crossThreadCopy(m_errorMessage); }
    enum class Error : uint8_t {
        Default = 0,
        OutOfMemory,
        Parse
    };
    Error error() const { return m_error; }

    bool WARN_UNUSED_RETURN failed() const { return !m_errorMessage.isNull(); }
    virtual bool hasWork() const = 0;
    enum CompilationEffort { All, Partial };
    virtual void work(CompilationEffort = All) = 0;
    virtual bool multiThreaded() const = 0;

    void waitForCompletion();
    // Returns true if it cancelled the plan.
    bool tryRemoveContextAndCancelIfLast(VM&);

protected:
    void runCompletionTasks() WTF_REQUIRES_LOCK(m_lock);
    void fail(String&& errorMessage, Error = Error::Default) WTF_REQUIRES_LOCK(m_lock);

    virtual bool isComplete() const = 0;
    virtual void complete() WTF_REQUIRES_LOCK(m_lock) = 0;

    CString signpostMessage(CompilationMode, uint32_t functionIndexSpace) const;
    void beginCompilerSignpost(CompilationMode, uint32_t functionIndexSpace) const;
    void beginCompilerSignpost(const Callee&) const;
    void endCompilerSignpost(CompilationMode, uint32_t functionIndexSpace) const;
    void endCompilerSignpost(const Callee&) const;

    MemoryMode m_mode { MemoryMode::BoundsChecking };
    Lock m_lock;
    Condition m_completed;

    Ref<ModuleInformation> m_moduleInformation;

    Vector<std::pair<VM*, CompletionTask>, 1> m_completionTasks;

    String m_errorMessage;
    Error m_error { Error::Default };
};


} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
