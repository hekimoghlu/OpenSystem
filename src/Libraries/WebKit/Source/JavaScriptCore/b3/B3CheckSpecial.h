/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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

#if ENABLE(B3_JIT)

#include "AirKind.h"
#include "B3StackmapSpecial.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace B3 {

namespace Air {
struct Inst;
}

// We want to lower Check instructions to a branch, but then we want to route that branch to our
// out-of-line code instead of doing anything else. For this reason, a CheckSpecial will remember
// which branch opcode we have selected along with the number of args in the overload we want. It
// will create an Inst with that opcode plus the appropriate args from the owning Inst whenever you
// call any of the callbacks.
//
// Note that for CheckAdd, CheckSub, and CheckMul we expect that the B3 arguments are the reverse
// of the Air arguments (Add(a, b) => Add32 b, a). Except:
// - CheckSub(0, x), which turns into BranchNeg32 x.
// - CheckMul(a, b), which turns into Mul32 b, a but we pass Any for a's ValueRep.

class CheckSpecial final : public StackmapSpecial {
    WTF_MAKE_TZONE_ALLOCATED(CheckSpecial);
public:
    // Support for hash consing these things.
    class Key {
    public:
        Key()
            : m_stackmapRole(SameAsRep)
            , m_numArgs(0)
        {
        }
        
        Key(Air::Kind kind, unsigned numArgs, RoleMode stackmapRole = SameAsRep)
            : m_kind(kind)
            , m_stackmapRole(stackmapRole)
            , m_numArgs(numArgs)
        {
        }

        explicit Key(const Air::Inst&);

        bool operator==(const Key& other) const
        {
            return m_kind == other.m_kind
                && m_numArgs == other.m_numArgs
                && m_stackmapRole == other.m_stackmapRole;
        }

        explicit operator bool() const { return *this != Key(); }

        Air::Kind kind() const { return m_kind; }
        unsigned numArgs() const { return m_numArgs; }
        RoleMode stackmapRole() const { return m_stackmapRole; }

        void dump(PrintStream& out) const;

        Key(WTF::HashTableDeletedValueType)
            : m_stackmapRole(SameAsRep)
            , m_numArgs(1)
        {
        }

        bool isHashTableDeletedValue() const
        {
            return *this == Key(WTF::HashTableDeletedValue);
        }

        unsigned hash() const
        {
            // Seriously, we don't need to be smart here. It just doesn't matter.
            return m_kind.hash() + m_numArgs + m_stackmapRole;
        }
        
    private:
        Air::Kind m_kind;
        RoleMode m_stackmapRole;
        unsigned m_numArgs;
    };
    
    CheckSpecial(Air::Kind, unsigned numArgs, RoleMode stackmapRole = SameAsRep);
    CheckSpecial(const Key&);
    ~CheckSpecial() final;

private:
    // Constructs and returns the Inst representing the branch that this will use.
    Air::Inst hiddenBranch(const Air::Inst&) const;

    void forEachArg(Air::Inst&, const ScopedLambda<Air::Inst::EachArgCallback>&) final;
    bool isValid(Air::Inst&) final;
    bool admitsStack(Air::Inst&, unsigned argIndex) final;
    bool admitsExtendedOffsetAddr(Air::Inst&, unsigned) final;
    std::optional<unsigned> shouldTryAliasingDef(Air::Inst&) final;

    // NOTE: the generate method will generate the hidden branch and then register a LatePath that
    // generates the stackmap. Super crazy dude!

    MacroAssembler::Jump generate(Air::Inst&, CCallHelpers&, Air::GenerationContext&) final;

    void dumpImpl(PrintStream&) const final;
    void deepDumpImpl(PrintStream&) const final;

    Air::Kind m_checkKind;
    RoleMode m_stackmapRole;
    unsigned m_numCheckArgs;
};

struct CheckSpecialKeyHash {
    static unsigned hash(const CheckSpecial::Key& key) { return key.hash(); }
    static bool equal(const CheckSpecial::Key& a, const CheckSpecial::Key& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} } // namespace JSC::B3

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::B3::CheckSpecial::Key> : JSC::B3::CheckSpecialKeyHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::B3::CheckSpecial::Key> : SimpleClassHashTraits<JSC::B3::CheckSpecial::Key> {
    // I don't want to think about this very hard, it's not worth it. I'm a be conservative.
    static constexpr bool emptyValueIsZero = false;
};

} // namespace WTF

#endif // ENABLE(B3_JIT)
