/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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

#if ENABLE(FTL_JIT)

#include "MacroAssemblerCodeRef.h"
#include "RegisterSet.h"

namespace JSC { namespace FTL {

// This is used for creating some sanity in slow-path calls out of the FTL's inline
// caches. The idea is that we don't want all of the register save/restore stuff to
// be generated at each IC site. Instead, the IC slow path call site will just save
// the registers needed for the arguments. It will arrange for there to be enough
// space on top of stack to save the remaining registers and the return PC. Then it
// will call a shared thunk that will save the remaining registers. That thunk needs
// to know the stack offset at which things get saved along with the call target.

// Note that the offset is *not including* the return PC that would be pushed on X86.

class SlowPathCallKey {
public:
    // Keep it within 2 bits.
    enum class Type : uint8_t {
        // For HashTables.
        Empty,
        Deleted,

        Direct,
        Indirect,
    };

    SlowPathCallKey() = default;
    
    SlowPathCallKey(const ScalarRegisterSet& set, CodePtr<CFunctionPtrTag> callTarget, uint8_t numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled, size_t offset, int32_t indirectOffset)
        : m_numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled(numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled)
        , m_offset(offset)
        , m_usedRegisters(set)
    {
        if (callTarget) {
            m_type = static_cast<unsigned>(Type::Direct);
            ASSERT(Type::Direct == this->type());
            m_callTarget = callTarget.retagged<OperationPtrTag>();
            ASSERT(!indirectOffset);
        } else {
            m_type = static_cast<unsigned>(Type::Indirect);
            ASSERT(Type::Indirect == this->type());
            m_indirectOffset = indirectOffset;
        }
        ASSERT(offset == m_offset);
    }
    
    CodePtr<OperationPtrTag> callTarget() const
    {
        if (type() == Type::Direct)
            return m_callTarget;
        return nullptr;
    }
    size_t offset() const { return m_offset; }
    const ScalarRegisterSet& usedRegisters() const { return m_usedRegisters; }
    RegisterSet argumentRegistersIfClobberingCheckIsEnabled() const
    {
        RELEASE_ASSERT(Options::clobberAllRegsInFTLICSlowPath());
        RegisterSet argumentRegisters;
        for (uint8_t i = 0; i < numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled(); ++i)
            argumentRegisters.add(GPRInfo::toArgumentRegister(i), IgnoreVectors);
        return argumentRegisters;
    }
    int32_t indirectOffset() const
    {
        if (type() == Type::Indirect)
            return m_indirectOffset;
        return 0;
    }
    
    SlowPathCallKey withCallTarget(CodePtr<CFunctionPtrTag> callTarget)
    {
        return SlowPathCallKey(usedRegisters(), callTarget, numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled(), offset(), indirectOffset());
    }
    
    void dump(PrintStream&) const;
    
    enum EmptyValueTag { EmptyValue };
    enum DeletedValueTag { DeletedValue };
    
    SlowPathCallKey(EmptyValueTag)
        : m_type(static_cast<unsigned>(Type::Empty))
    {
        ASSERT(Type::Empty == this->type());
    }
    
    SlowPathCallKey(DeletedValueTag)
        : m_type(static_cast<unsigned>(Type::Deleted))
    {
        ASSERT(Type::Deleted == this->type());
    }
    
    bool isEmptyValue() const { return type() == Type::Empty; }
    bool isDeletedValue() const { return type() == Type::Deleted; }
    
    bool operator==(const SlowPathCallKey& other) const
    {
        return m_offset == other.m_offset
            && m_usedRegisters == other.m_usedRegisters
            && numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled() == other.numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled()
            && type() == other.type()
            && callTarget() == other.callTarget()
            && indirectOffset() == other.indirectOffset();
    }
    unsigned hash() const
    {
        // m_numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled is intentionally not included because it will always be 0
        // unless Options::clobberAllRegsInFTLICSlowPath() is set, and Options::clobberAllRegsInFTLICSlowPath() is only set in debugging use cases.
        return PtrHash<void*>::hash(callTarget().taggedPtr()) + m_offset + m_usedRegisters.hash() + indirectOffset() + static_cast<unsigned>(type());
    }

private:
    static_assert(NUMBER_OF_ARGUMENT_REGISTERS <= std::numeric_limits<uint8_t>::max());

    uint8_t numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled() const { return m_numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled; }
    Type type() const { return static_cast<Type>(m_type); }

    union {
        CodePtr<OperationPtrTag> m_callTarget { };
        int32_t m_indirectOffset;
    };
    size_t m_numberOfUsedArgumentRegistersIfClobberingCheckIsEnabled : 8 { 0 };
    size_t m_type : 2 { static_cast<size_t>(Type::Empty) };
    size_t m_offset : 54 { 0 };
    ScalarRegisterSet m_usedRegisters;
};


struct SlowPathCallKeyHash {
    static unsigned hash(const SlowPathCallKey& key) { return key.hash(); }
    static bool equal(const SlowPathCallKey& a, const SlowPathCallKey& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

} } // namespace JSC::FTL

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::FTL::SlowPathCallKey> : JSC::FTL::SlowPathCallKeyHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::FTL::SlowPathCallKey> : public CustomHashTraits<JSC::FTL::SlowPathCallKey> { };

} // namespace WTF

#endif // ENABLE(FTL_JIT)
