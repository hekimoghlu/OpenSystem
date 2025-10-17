/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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

#if ENABLE(JIT)

#include "CompilationResult.h"
#include "JITCode.h"
#include "JITCompilationKey.h"
#include "JITCompilationMode.h"
#include "JITPlanStage.h"
#include "ReleaseHeapAccessScope.h"
#include <wtf/MonotonicTime.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace JSC {

class AbstractSlotVisitor;
class CodeBlock;
class JITWorklistThread;
class VM;

class JITPlan : public ThreadSafeRefCounted<JITPlan> {
protected:
    JITPlan(JITCompilationMode, CodeBlock*);

public:
    virtual ~JITPlan();

    VM* vm() const { return m_vm; }
    CodeBlock* codeBlock() const { return m_codeBlock; }
    JITWorklistThread* thread() const { return m_thread; }

    JITCompilationMode mode() const { return m_mode; }

    JITPlanStage stage() const { return m_stage; }
    bool isDFG() const { return ::JSC::isDFG(m_mode); }
    bool isFTL() const { return ::JSC::isFTL(m_mode); }
    bool isUnlinked() const { return ::JSC::isUnlinked(m_mode); }

    enum class Tier { Baseline = 0, DFG = 1, FTL = 2, Count = 3 };
    Tier tier() const;
    JITType jitType() const
    {
        switch (tier()) {
        case Tier::Baseline:
            return JITType::BaselineJIT;
        case Tier::DFG:
            return JITType::DFGJIT;
        case Tier::FTL:
            return JITType::FTLJIT;
        default:
            return JITType::None;
        }
    }

    JITCompilationKey key();

    void compileInThread(JITWorklistThread*);

    virtual size_t codeSize() const = 0;

    virtual CompilationResult finalize() = 0;

    virtual void finalizeInGC() { }

    void notifyCompiling();
    virtual void notifyReady();
    virtual void cancel();

    virtual bool isKnownToBeLiveAfterGC();
    virtual bool isKnownToBeLiveDuringGC(AbstractSlotVisitor&);
    virtual bool iterateCodeBlocksForGC(AbstractSlotVisitor&, const Function<void(CodeBlock*)>&);
    virtual bool checkLivenessAndVisitChildren(AbstractSlotVisitor&);

    bool isInSafepoint() const;
    bool safepointKeepsDependenciesLive() const;

    template<typename Functor>
    void addMainThreadFinalizationTask(const Functor& functor)
    {
        m_mainThreadFinalizationTasks.append(createSharedTask<void()>(functor));
    }

    void runMainThreadFinalizationTasks();

protected:
    bool computeCompileTimes() const;
    bool reportCompileTimes() const;

    enum CompilationPath { FailPath, BaselinePath, DFGPath, FTLPath, CancelPath };
    virtual CompilationPath compileInThreadImpl() = 0;

    JITPlanStage m_stage { JITPlanStage::Preparing };
    JITCompilationMode m_mode;
    MonotonicTime m_timeBeforeFTL;
    VM* m_vm;
    CodeBlock* m_codeBlock;
    JITWorklistThread* m_thread { nullptr };
    Vector<RefPtr<SharedTask<void()>>> m_mainThreadFinalizationTasks;
};

} // namespace JSC

#endif // ENABLE(JIT)
