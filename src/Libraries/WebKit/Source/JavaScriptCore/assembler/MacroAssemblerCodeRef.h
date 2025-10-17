/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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

#include "ExecutableMemoryHandle.h"
#include "JSCPtrTag.h"
#include <wtf/CodePtr.h>
#include <wtf/DataLog.h>
#include <wtf/PrintStream.h>
#include <wtf/RefPtr.h>
#include <wtf/text/CString.h>

#if CPU(ARM_THUMB2) && ENABLE(JIT)
// ARM instructions must be 16-bit aligned. Thumb2 code pointers to be loaded into
// into the processor are decorated with the bottom bit set, while traditional ARM has
// the lower bit clear. Since we don't know what kind of pointer, we check for both
// decorated and undecorated null.
#define ASSERT_VALID_CODE_OFFSET(offset) \
    ASSERT(!(offset & 1)) // Must be multiple of 2.
#else
#define ASSERT_VALID_CODE_OFFSET(offset) // Anything goes!
#endif

namespace JSC {

namespace Wasm {
enum class CompilationMode : uint8_t;
} // namespace Wasm

class CodeBlock;

enum OpcodeID : unsigned;

// MacroAssemblerCodeRef:
//
// A reference to a section of JIT generated code.  A CodeRef consists of a
// pointer to the code, and a ref pointer to the pool from within which it
// was allocated.
class MacroAssemblerCodeRefBase {
protected:
    static bool tryToDisassemble(CodePtr<DisassemblyPtrTag>, size_t, const char* prefix, PrintStream& out);
    static bool tryToDisassemble(CodePtr<DisassemblyPtrTag>, size_t, const char* prefix);
    JS_EXPORT_PRIVATE static CString disassembly(CodePtr<DisassemblyPtrTag>, size_t);
};

template<PtrTag tag>
class MacroAssemblerCodeRef : private MacroAssemblerCodeRefBase {
private:
    // This is private because it's dangerous enough that we want uses of it
    // to be easy to find - hence the static create method below.
    explicit MacroAssemblerCodeRef(CodePtr<tag> codePtr)
        : m_codePtr(codePtr)
    {
        ASSERT(m_codePtr);
    }

public:
    MacroAssemblerCodeRef() = default;

    MacroAssemblerCodeRef(Ref<ExecutableMemoryHandle>&& executableMemory)
        : m_codePtr(executableMemory->start().retaggedPtr<tag>())
        , m_executableMemory(WTFMove(executableMemory))
    {
        ASSERT(m_executableMemory->start());
        ASSERT(m_codePtr);
    }

    template<PtrTag otherTag>
    MacroAssemblerCodeRef& operator=(const MacroAssemblerCodeRef<otherTag>& otherCodeRef)
    {
        m_codePtr = CodePtr<tag>::fromTaggedPtr(otherCodeRef.code().template retaggedPtr<tag>());
        m_executableMemory = otherCodeRef.m_executableMemory;
        return *this;
    }
    
    // Use this only when you know that the codePtr refers to code that is
    // already being kept alive through some other means. Typically this means
    // that codePtr is immortal.
    static MacroAssemblerCodeRef createSelfManagedCodeRef(CodePtr<tag> codePtr)
    {
        return MacroAssemblerCodeRef(codePtr);
    }
    
    ExecutableMemoryHandle* executableMemory() const
    {
        return m_executableMemory.get();
    }
    
    CodePtr<tag> code() const
    {
        return m_codePtr;
    }

    template<PtrTag newTag>
    CodePtr<newTag> retaggedCode() const
    {
        return m_codePtr.template retagged<newTag>();
    }

    template<PtrTag newTag>
    MacroAssemblerCodeRef<newTag> retagged() const
    {
        return MacroAssemblerCodeRef<newTag>(*this);
    }

    size_t size() const
    {
        if (!m_executableMemory)
            return 0;
        return m_executableMemory->sizeInBytes();
    }

    bool tryToDisassemble(PrintStream& out, const char* prefix = "") const
    {
        return tryToDisassemble(retaggedCode<DisassemblyPtrTag>(), size(), prefix, out);
    }
    
    bool tryToDisassemble(const char* prefix = "") const
    {
        return tryToDisassemble(retaggedCode<DisassemblyPtrTag>(), size(), prefix);
    }
    
    CString disassembly() const
    {
        return MacroAssemblerCodeRefBase::disassembly(retaggedCode<DisassemblyPtrTag>(), size());
    }
    
    explicit operator bool() const { return !!m_codePtr; }
    
    void dump(PrintStream& out) const
    {
        m_codePtr.dumpWithName("CodeRef"_s, out);
    }

    static constexpr ptrdiff_t offsetOfCodePtr() { return OBJECT_OFFSETOF(MacroAssemblerCodeRef, m_codePtr); }

private:
    template<PtrTag otherTag>
    MacroAssemblerCodeRef(const MacroAssemblerCodeRef<otherTag>& otherCodeRef)
    {
        *this = otherCodeRef;
    }

    CodePtr<tag> m_codePtr;
    RefPtr<ExecutableMemoryHandle> m_executableMemory;

    template<PtrTag> friend class MacroAssemblerCodeRef;
};

bool shouldDumpDisassemblyFor(CodeBlock*);
bool shouldDumpDisassemblyFor(Wasm::CompilationMode);

} // namespace JSC
