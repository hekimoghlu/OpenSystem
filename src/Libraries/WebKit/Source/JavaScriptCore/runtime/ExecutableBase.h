/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

#include "ArityCheckMode.h"
#include "CallData.h"
#include "CodeBlockHash.h"
#include "CodeSpecializationKind.h"
#include "JITCode.h"
#include "UnlinkedCodeBlock.h"
#include "UnlinkedFunctionExecutable.h"

namespace JSC {

class CodeBlock;
class EvalCodeBlock;
class FunctionCodeBlock;
class JSScope;
class JSWebAssemblyModule;
class LLIntOffsetsExtractor;
class ModuleProgramCodeBlock;
class ProgramCodeBlock;

enum CompilationKind { FirstCompilation, OptimizingCompilation };

inline bool isCall(CodeSpecializationKind kind)
{
    if (kind == CodeForCall)
        return true;
    ASSERT(kind == CodeForConstruct);
    return false;
}

class ExecutableBase : public JSCell {
    friend class JIT;
    friend class LLIntOffsetsExtractor;
    friend MacroAssemblerCodeRef<JSEntryPtrTag> boundFunctionCallGenerator(VM*);
    using Base = JSCell;

protected:
    ExecutableBase(VM& vm, Structure* structure)
        : JSCell(vm, structure)
    {
    }

    DECLARE_DEFAULT_FINISH_CREATION;

public:
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);
    
    // Force subclasses to override this.
    template<typename, SubspaceAccess>
    static void subspaceFor(VM&) { }
        
    CodeBlockHash hashFor(CodeSpecializationKind) const;

    bool isEvalExecutable() const
    {
        return type() == EvalExecutableType;
    }
    bool isFunctionExecutable() const
    {
        return type() == FunctionExecutableType;
    }
    bool isProgramExecutable() const
    {
        return type() == ProgramExecutableType;
    }
    bool isModuleProgramExecutable()
    {
        return type() == ModuleProgramExecutableType;
    }
    bool isHostFunction() const
    {
        return type() == NativeExecutableType;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_EXPORT_INFO;

public:
    Ref<JSC::JITCode> generatedJITCodeForCall() const
    {
        ASSERT(m_jitCodeForCall);
        return *m_jitCodeForCall;
    }

    Ref<JSC::JITCode> generatedJITCodeForConstruct() const
    {
        ASSERT(m_jitCodeForConstruct);
        return *m_jitCodeForConstruct;
    }
        
    Ref<JSC::JITCode> generatedJITCodeFor(CodeSpecializationKind kind) const
    {
        if (kind == CodeForCall)
            return generatedJITCodeForCall();
        ASSERT(kind == CodeForConstruct);
        return generatedJITCodeForConstruct();
    }

    CodePtr<JSEntryPtrTag> generatedJITCodeWithArityCheckForCall() const
    {
        return m_jitCodeForCallWithArityCheck;
    }

    CodePtr<JSEntryPtrTag> generatedJITCodeWithArityCheckForConstruct() const
    {
        return m_jitCodeForConstructWithArityCheck;
    }

    CodePtr<JSEntryPtrTag> generatedJITCodeWithArityCheckFor(CodeSpecializationKind kind) const
    {
        if (kind == CodeForCall)
            return generatedJITCodeWithArityCheckForCall();
        ASSERT(kind == CodeForConstruct);
        return generatedJITCodeWithArityCheckForConstruct();
    }

    CodePtr<JSEntryPtrTag> entrypointFor(CodeSpecializationKind kind, ArityCheckMode arity)
    {
        // Check if we have a cached result. We only have it for arity check because we use the
        // no-arity entrypoint in non-virtual calls, which will "cache" this value directly in
        // machine code.
        if (arity == MustCheckArity) {
            switch (kind) {
            case CodeForCall:
                if (CodePtr<JSEntryPtrTag> result = m_jitCodeForCallWithArityCheck)
                    return result;
                break;
            case CodeForConstruct:
                if (CodePtr<JSEntryPtrTag> result = m_jitCodeForConstructWithArityCheck)
                    return result;
                break;
            }
        }
        CodePtr<JSEntryPtrTag> result = generatedJITCodeFor(kind)->addressForCall(arity);
        if (arity == MustCheckArity) {
            // Cache the result; this is necessary for the JIT's virtual call optimizations.
            switch (kind) {
            case CodeForCall:
                m_jitCodeForCallWithArityCheck = result;
                break;
            case CodeForConstruct:
                m_jitCodeForConstructWithArityCheck = result;
                break;
            }
        }
        return result;
    }

    static constexpr ptrdiff_t offsetOfJITCodeWithArityCheckFor(
        CodeSpecializationKind kind)
    {
        switch (kind) {
        case CodeForCall:
            return OBJECT_OFFSETOF(ExecutableBase, m_jitCodeForCallWithArityCheck);
        case CodeForConstruct:
            return OBJECT_OFFSETOF(ExecutableBase, m_jitCodeForConstructWithArityCheck);
        }
        RELEASE_ASSERT_NOT_REACHED();
        return 0;
    }
    
    bool hasJITCodeForCall() const;
    bool hasJITCodeForConstruct() const;

    bool hasJITCodeFor(CodeSpecializationKind kind) const
    {
        if (kind == CodeForCall)
            return hasJITCodeForCall();
        ASSERT(kind == CodeForConstruct);
        return hasJITCodeForConstruct();
    }

    // Intrinsics are only for calls, currently.
    inline Intrinsic intrinsic() const;
        
    inline Intrinsic intrinsicFor(CodeSpecializationKind) const;

    ImplementationVisibility implementationVisibility() const;
    InlineAttribute inlineAttribute() const;

    CodePtr<JSEntryPtrTag> swapGeneratedJITCodeWithArityCheckForDebugger(CodeSpecializationKind kind, CodePtr<JSEntryPtrTag> jitCodeWithArityCheck)
    {
        if (kind == CodeForCall)
            return swapGeneratedJITCodeForCallWithArityCheckForDebugger(jitCodeWithArityCheck);
        ASSERT(kind == CodeForConstruct);
        return swapGeneratedJITCodeForConstructWithArityCheckForDebugger(jitCodeWithArityCheck);
    }

    CodePtr<JSEntryPtrTag> swapGeneratedJITCodeForCallWithArityCheckForDebugger(CodePtr<JSEntryPtrTag> jitCodeForCallWithArityCheck)
    {
        auto old = m_jitCodeForCallWithArityCheck;
        m_jitCodeForCallWithArityCheck = jitCodeForCallWithArityCheck;
        return old;
    }

    CodePtr<JSEntryPtrTag> swapGeneratedJITCodeForConstructWithArityCheckForDebugger(CodePtr<JSEntryPtrTag> jitCodeForConstructWithArityCheck)
    {
        auto old = m_jitCodeForConstructWithArityCheck;
        m_jitCodeForConstructWithArityCheck = jitCodeForConstructWithArityCheck;
        return old;
    }
    
    void dump(PrintStream&) const;
        
protected:
    RefPtr<JSC::JITCode> m_jitCodeForCall;
    RefPtr<JSC::JITCode> m_jitCodeForConstruct;
    CodePtr<JSEntryPtrTag> m_jitCodeForCallWithArityCheck;
    CodePtr<JSEntryPtrTag> m_jitCodeForConstructWithArityCheck;
};

} // namespace JSC
