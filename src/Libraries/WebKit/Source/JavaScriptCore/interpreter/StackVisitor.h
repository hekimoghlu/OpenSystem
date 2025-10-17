/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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

#include "BytecodeIndex.h"
#include "CalleeBits.h"
#include "LineColumn.h"
#include "SourceID.h"
#include "WasmIndexOrName.h"
#include <wtf/Function.h>
#include <wtf/Indenter.h>
#include <wtf/IterationStatus.h>
#include <wtf/text/WTFString.h>

namespace JSC {

struct EntryFrame;
struct InlineCallFrame;

class CallFrame;
class CodeBlock;
class CodeOrigin;
class JSCell;
class JSFunction;
class ClonedArguments;
class Register;
class RegisterAtOffsetList;

class StackVisitor {
public:
    class Frame {
    public:
        enum CodeType {
            Global,
            Eval,
            Function,
            Module,
            Native,
            Wasm
        };

        size_t index() const { return m_index; }
        size_t argumentCountIncludingThis() const { return m_argumentCountIncludingThis; }
        bool callerIsEntryFrame() const { return m_callerIsEntryFrame; }
        CallFrame* callerFrame() const { return m_callerFrame; }
        EntryFrame* entryFrame() const { return m_entryFrame; }
        CalleeBits callee() const { return m_callee; }
        CodeBlock* codeBlock() const { return m_codeBlock; }
        BytecodeIndex bytecodeIndex() const { return m_bytecodeIndex; }
        InlineCallFrame* inlineCallFrame() const {
#if ENABLE(DFG_JIT)
            return m_inlineDFGCallFrame;
#else
            return nullptr;
#endif
        }
        void* returnPC() const { return m_returnPC; }

        bool isNativeFrame() const { return !codeBlock() && !isNativeCalleeFrame(); }
        bool isInlinedDFGFrame() const { return !isNativeCalleeFrame() && !!inlineCallFrame(); }
        bool isNativeCalleeFrame() const { return m_callee.isNativeCallee(); }
        Wasm::IndexOrName const wasmFunctionIndexOrName()
        {
            ASSERT(isNativeCalleeFrame());
            return m_wasmFunctionIndexOrName;
        }

        JS_EXPORT_PRIVATE String functionName() const;
        JS_EXPORT_PRIVATE String sourceURL() const;
        JS_EXPORT_PRIVATE String preRedirectURL() const;
        JS_EXPORT_PRIVATE String toString() const;

        JS_EXPORT_PRIVATE SourceID sourceID();

        CodeType codeType() const;
        bool hasLineAndColumnInfo() const;
        JS_EXPORT_PRIVATE LineColumn computeLineAndColumn() const;

#if ENABLE(ASSEMBLER)
        std::optional<RegisterAtOffsetList> calleeSaveRegistersForUnwinding();
#endif

        ClonedArguments* createArguments(VM&);
        CallFrame* callFrame() const { return m_callFrame; }

        JS_EXPORT_PRIVATE bool isImplementationVisibilityPrivate() const;

        void dump(PrintStream&, Indenter = Indenter()) const;
        void dump(PrintStream&, Indenter, WTF::Function<void(PrintStream&)> prefix) const;

    private:
        Frame() { }
        ~Frame() { }

        void setToEnd();

#if ENABLE(DFG_JIT)
        InlineCallFrame* m_inlineDFGCallFrame { nullptr };
#endif
        unsigned m_wasmDistanceFromDeepestInlineFrame { 0 };
        CallFrame* m_callFrame { nullptr };
        EntryFrame* m_entryFrame { nullptr };
        EntryFrame* m_callerEntryFrame { nullptr };
        CallFrame* m_callerFrame { nullptr };
        CalleeBits m_callee { };
        CodeBlock* m_codeBlock { nullptr };
        void* m_returnPC { nullptr };
        size_t m_index { 0 };
        size_t m_argumentCountIncludingThis { 0 };
        BytecodeIndex m_bytecodeIndex { };
        bool m_callerIsEntryFrame : 1 { false };
        bool m_isWasmFrame : 1 { false };
        Wasm::IndexOrName m_wasmFunctionIndexOrName { };

        friend class StackVisitor;
    };

    // StackVisitor::visit() expects a Functor that implements the following method:
    //     IterationStatus operator()(StackVisitor&) const;

    enum EmptyEntryFrameAction {
        ContinueIfTopEntryFrameIsEmpty,
        TerminateIfTopEntryFrameIsEmpty,
    };

    template <EmptyEntryFrameAction action = ContinueIfTopEntryFrameIsEmpty, typename Functor>
    static void visit(CallFrame* startFrame, VM& vm, const Functor& functor, bool skipFirstFrame = false)
    {
        StackVisitor visitor(startFrame, vm, skipFirstFrame);
        if (action == TerminateIfTopEntryFrameIsEmpty && visitor.topEntryFrameIsEmpty())
            return;
        while (visitor->callFrame()) {
            IterationStatus status = functor(visitor);
            if (status != IterationStatus::Continue)
                break;
            visitor.gotoNextFrame();
        }
    }

    Frame& operator*() { return m_frame; }
    ALWAYS_INLINE Frame* operator->() { return &m_frame; }
    void unwindToMachineCodeBlockFrame();

    bool topEntryFrameIsEmpty() const { return m_topEntryFrameIsEmpty; }

private:
    JS_EXPORT_PRIVATE StackVisitor(CallFrame* startFrame, VM&, bool skipFirstFrame);

    JS_EXPORT_PRIVATE void gotoNextFrame();

    void readFrame(CallFrame*);
    void readInlinableNativeCalleeFrame(CallFrame*);
    void readNonInlinedFrame(CallFrame*, CodeOrigin* = nullptr);
#if ENABLE(DFG_JIT)
    void readInlinedFrame(CallFrame*, CodeOrigin*);
#endif
    CallFrame* updatePreviousReturnPCIfNecessary(CallFrame*);

    Frame m_frame;
    void* m_previousReturnPC { nullptr };
    bool m_topEntryFrameIsEmpty { false };
};

class CallerFunctor {
public:
    CallerFunctor()
        : m_hasSkippedFirstFrame(false)
        , m_callerFrame(nullptr)
    {
    }

    CallFrame* callerFrame() const { return m_callerFrame; }

    IterationStatus operator()(StackVisitor& visitor) const
    {
        if (!m_hasSkippedFirstFrame) {
            m_hasSkippedFirstFrame = true;
            return IterationStatus::Continue;
        }

        m_callerFrame = visitor->callFrame();
        return IterationStatus::Done;
    }

private:
    mutable bool m_hasSkippedFirstFrame;
    mutable CallFrame* m_callerFrame;
};

} // namespace JSC
