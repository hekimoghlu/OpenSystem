/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

#if ENABLE(SAMPLING_PROFILER)

#include "CallFrame.h"
#include "CodeBlockHash.h"
#include "JITCode.h"
#include "MachineStackMarker.h"
#include "NativeCallee.h"
#include "PCToCodeOriginMap.h"
#include "WasmCompilationMode.h"
#include "WasmIndexOrName.h"
#include <wtf/Box.h>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/Stopwatch.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRandom.h>

namespace JSC {

class VM;
class ExecutableBase;

class SamplingProfiler : public ThreadSafeRefCounted<SamplingProfiler> {
    WTF_MAKE_TZONE_ALLOCATED(SamplingProfiler);
public:

    struct UnprocessedStackFrame {
        UnprocessedStackFrame(CodeBlock* codeBlock, CalleeBits callee, CallSiteIndex callSiteIndex)
            : unverifiedCallee(callee)
            , verifiedCodeBlock(codeBlock)
            , callSiteIndex(callSiteIndex)
        { }

        UnprocessedStackFrame(const void* pc)
            : cCodePC(pc)
        { }

        UnprocessedStackFrame() = default;

        const void* cCodePC { nullptr };
        CalleeBits unverifiedCallee;
        CodeBlock* verifiedCodeBlock { nullptr };
        CallSiteIndex callSiteIndex;
        NativeCallee::Category nativeCalleeCategory { NativeCallee::Category::InlineCache };
#if ENABLE(WEBASSEMBLY)
        std::optional<Wasm::IndexOrName> wasmIndexOrName;
#endif
        std::optional<Wasm::CompilationMode> wasmCompilationMode;
#if ENABLE(JIT)
        Box<PCToCodeOriginMap> wasmPCMap;
#endif
    };

    enum class FrameType { 
        Executable,
        Wasm,
        Host,
        RegExp,
        C,
        Unknown,
    };

    struct StackFrame {
        StackFrame(ExecutableBase* executable)
            : frameType(FrameType::Executable)
            , executable(executable)
        { }

        StackFrame()
        { }

        FrameType frameType { FrameType::Unknown };
        const void* cCodePC { nullptr };
        ExecutableBase* executable { nullptr };
        JSObject* callee { nullptr };
        RegExp* regExp { nullptr };
#if ENABLE(WEBASSEMBLY)
        std::optional<Wasm::IndexOrName> wasmIndexOrName;
#endif
        std::optional<Wasm::CompilationMode> wasmCompilationMode;
        BytecodeIndex wasmOffset;

        struct CodeLocation {
            bool hasCodeBlockHash() const
            {
                return codeBlockHash.isSet();
            }

            bool hasBytecodeIndex() const
            {
                return !!bytecodeIndex;
            }

            bool hasExpressionInfo() const
            {
                return lineColumn.line != std::numeric_limits<unsigned>::max()
                    && lineColumn.column != std::numeric_limits<unsigned>::max();
            }

            // These attempt to be expression-level line and column number.
            LineColumn lineColumn { std::numeric_limits<unsigned>::max(), std::numeric_limits<unsigned>::max() };
            BytecodeIndex bytecodeIndex;
            CodeBlockHash codeBlockHash;
            JITType jitType { JITType::None };
            bool isRegExp { false };
        };

        CodeLocation semanticLocation;
        std::optional<std::pair<CodeLocation, CodeBlock*>> machineLocation; // This is non-null if we were inlined. It represents the machine frame we were inlined into.

        bool hasExpressionInfo() const { return semanticLocation.hasExpressionInfo(); }
        unsigned lineNumber() const
        {
            ASSERT(hasExpressionInfo());
            return semanticLocation.lineColumn.line;
        }
        unsigned columnNumber() const
        {
            ASSERT(hasExpressionInfo());
            return semanticLocation.lineColumn.column;
        }

        // These are function-level data.
        String nameFromCallee(VM&);
        String displayName(VM&);
        int functionStartLine();
        unsigned functionStartColumn();
        std::tuple<SourceProvider*, SourceID> sourceProviderAndID();
        String url();
    };

    struct UnprocessedStackTrace {
        MonotonicTime timestamp;
        Seconds stopwatchTimestamp;
        void* topPC;
        bool topFrameIsLLInt;
        void* llintPC;
        RegExp* regExp;
        Vector<UnprocessedStackFrame> frames;
    };

    struct StackTrace {
        MonotonicTime timestamp;
        Seconds stopwatchTimestamp;
        Vector<StackFrame> frames;
        StackTrace()
        { }
        StackTrace(StackTrace&& other)
            : timestamp(other.timestamp)
            , frames(WTFMove(other.frames))
        { }
    };

    SamplingProfiler(VM&, Ref<Stopwatch>&&);
    ~SamplingProfiler();
    void noticeJSLockAcquisition();
    void noticeVMEntry();
    void shutdown();
    template<typename Visitor> void visit(Visitor&) WTF_REQUIRES_LOCK(m_lock);
    Lock& getLock() WTF_RETURNS_LOCK(m_lock) { return m_lock; }
    void setTimingInterval(Seconds interval) { m_timingInterval = interval; }
    JS_EXPORT_PRIVATE void start();
    void startWithLock() WTF_REQUIRES_LOCK(m_lock);
    Vector<StackTrace> releaseStackTraces() WTF_REQUIRES_LOCK(m_lock);
    JS_EXPORT_PRIVATE Ref<JSON::Value> stackTracesAsJSON();
    JS_EXPORT_PRIVATE void noticeCurrentThreadAsJSCExecutionThread();
    void noticeCurrentThreadAsJSCExecutionThreadWithLock() WTF_REQUIRES_LOCK(m_lock);
    void processUnverifiedStackTraces() WTF_REQUIRES_LOCK(m_lock);
    void setStopWatch(Ref<Stopwatch>&& stopwatch) WTF_REQUIRES_LOCK(m_lock) { m_stopwatch = WTFMove(stopwatch); }
    void pause() WTF_REQUIRES_LOCK(m_lock);
    void clearData() WTF_REQUIRES_LOCK(m_lock);

    // Used for debugging in the JSC shell/DRT.
    void registerForReportAtExit();
    void reportDataToOptionFile();
    JS_EXPORT_PRIVATE void reportTopFunctions();
    JS_EXPORT_PRIVATE void reportTopFunctions(PrintStream&);
    JS_EXPORT_PRIVATE void reportTopBytecodes();
    JS_EXPORT_PRIVATE void reportTopBytecodes(PrintStream&);

    JS_EXPORT_PRIVATE Thread* thread() const;

private:
    void createThreadIfNecessary() WTF_REQUIRES_LOCK(m_lock);
    void timerLoop();
    void takeSample(Seconds& stackTraceProcessingTime) WTF_REQUIRES_LOCK(m_lock);

    Lock m_lock;
    bool m_isPaused WTF_GUARDED_BY_LOCK(m_lock);
    bool m_isShutDown WTF_GUARDED_BY_LOCK(m_lock);
    bool m_needsReportAtExit { false };
    VM& m_vm;
    WeakRandom m_weakRandom;
    Ref<Stopwatch> m_stopwatch WTF_GUARDED_BY_LOCK(m_lock);
    Vector<StackTrace> m_stackTraces WTF_GUARDED_BY_LOCK(m_lock);
    Vector<UnprocessedStackTrace> m_unprocessedStackTraces WTF_GUARDED_BY_LOCK(m_lock);
    Seconds m_timingInterval;
    RefPtr<Thread> m_thread;
    RefPtr<Thread> m_jscExecutionThread WTF_GUARDED_BY_LOCK(m_lock);
    UncheckedKeyHashSet<JSCell*> m_liveCellPointers WTF_GUARDED_BY_LOCK(m_lock);
    Vector<UnprocessedStackFrame> m_currentFrames WTF_GUARDED_BY_LOCK(m_lock);
};

} // namespace JSC

namespace WTF {

void printInternal(PrintStream&, JSC::SamplingProfiler::FrameType);

} // namespace WTF

#endif // ENABLE(SAMPLING_PROFILER)
