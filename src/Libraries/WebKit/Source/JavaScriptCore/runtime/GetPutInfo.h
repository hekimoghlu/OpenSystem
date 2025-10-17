/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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

#include "ECMAMode.h"
#include <wtf/text/UniquedStringImpl.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class Structure;
class WatchpointSet;
class JSLexicalEnvironment;

enum ResolveMode {
    ThrowIfNotFound,
    DoNotThrowIfNotFound
};

#define FOR_EACH_RESOLVE_TYPE(v) \
    v(GlobalProperty) \
    v(GlobalVar) \
    v(GlobalLexicalVar) \
    v(ClosureVar) \
    v(ResolvedClosureVar) \
    v(ModuleVar) \
    v(GlobalPropertyWithVarInjectionChecks) \
    v(GlobalVarWithVarInjectionChecks) \
    v(GlobalLexicalVarWithVarInjectionChecks) \
    v(ClosureVarWithVarInjectionChecks) \
    v(UnresolvedProperty) \
    v(UnresolvedPropertyWithVarInjectionChecks) \
    v(Dynamic)

enum ResolveType : unsigned {
    // Lexical scope guaranteed a certain type of variable access.
    GlobalProperty,
    GlobalVar,
    GlobalLexicalVar,
    ClosureVar,
    ResolvedClosureVar,
    ModuleVar,

    // Ditto, but at least one intervening scope used non-strict eval, which
    // can inject an intercepting var delcaration at runtime.
    GlobalPropertyWithVarInjectionChecks,
    GlobalVarWithVarInjectionChecks,
    GlobalLexicalVarWithVarInjectionChecks,
    ClosureVarWithVarInjectionChecks,

    // We haven't found which scope this belongs to, and we also
    // haven't ruled out the possibility of it being cached. Ideally,
    // we want to transition this to GlobalVar/GlobalLexicalVar/GlobalProperty <with/without injection>
    UnresolvedProperty,
    UnresolvedPropertyWithVarInjectionChecks,

    // Lexical scope didn't prove anything -- probably because of a 'with' scope.
    Dynamic
};

enum class InitializationMode : unsigned {
    Initialization,      // "let x = 20;"
    ConstInitialization, // "const x = 20;"
    NotInitialization,   // "x = 20;"
    ScopedArgumentInitialization // Assign to scoped argument, which also has NotInitialization semantics.
};

ALWAYS_INLINE const char* resolveModeName(ResolveMode resolveMode)
{
    static const char* const names[] = {
        "ThrowIfNotFound",
        "DoNotThrowIfNotFound"
    };
    return names[resolveMode];
}

ALWAYS_INLINE const char* resolveTypeName(ResolveType type)
{
    static const char* const names[] = {
        "GlobalProperty",
        "GlobalVar",
        "GlobalLexicalVar",
        "ClosureVar",
        "ResolvedClosureVar",
        "ModuleVar",
        "GlobalPropertyWithVarInjectionChecks",
        "GlobalVarWithVarInjectionChecks",
        "GlobalLexicalVarWithVarInjectionChecks",
        "ClosureVarWithVarInjectionChecks",
        "UnresolvedProperty",
        "UnresolvedPropertyWithVarInjectionChecks",
        "Dynamic"
    };
    return names[type];
}

ALWAYS_INLINE const char* initializationModeName(InitializationMode initializationMode)
{
    static const char* const names[] = {
        "Initialization",
        "ConstInitialization",
        "NotInitialization",
        "ScopedArgumentInitialization"
    };
    return names[static_cast<unsigned>(initializationMode)];
}

ALWAYS_INLINE bool isInitialization(InitializationMode initializationMode)
{
    switch (initializationMode) {
    case InitializationMode::Initialization:
    case InitializationMode::ConstInitialization:
        return true;
    case InitializationMode::NotInitialization:
    case InitializationMode::ScopedArgumentInitialization:
        return false;
    }
    ASSERT_NOT_REACHED();
    return false;
}

ALWAYS_INLINE ResolveType makeType(ResolveType type, bool needsVarInjectionChecks)
{
    if (!needsVarInjectionChecks)
        return type;

    switch (type) {
    case GlobalProperty:
        return GlobalPropertyWithVarInjectionChecks;
    case GlobalVar:
        return GlobalVarWithVarInjectionChecks;
    case GlobalLexicalVar:
        return GlobalLexicalVarWithVarInjectionChecks;
    case ClosureVar:
    case ResolvedClosureVar:
        return ClosureVarWithVarInjectionChecks;
    case UnresolvedProperty:
        return UnresolvedPropertyWithVarInjectionChecks;
    case ModuleVar:
    case GlobalPropertyWithVarInjectionChecks:
    case GlobalVarWithVarInjectionChecks:
    case GlobalLexicalVarWithVarInjectionChecks:
    case ClosureVarWithVarInjectionChecks:
    case UnresolvedPropertyWithVarInjectionChecks:
    case Dynamic:
        return type;
    }

    RELEASE_ASSERT_NOT_REACHED();
    return type;
}

ALWAYS_INLINE bool needsVarInjectionChecks(ResolveType type)
{
    switch (type) {
    case GlobalProperty:
    case GlobalVar:
    case GlobalLexicalVar:
    case ClosureVar:
    case ResolvedClosureVar:
    case ModuleVar:
    case UnresolvedProperty:
        return false;
    case GlobalPropertyWithVarInjectionChecks:
    case GlobalVarWithVarInjectionChecks:
    case GlobalLexicalVarWithVarInjectionChecks:
    case ClosureVarWithVarInjectionChecks:
    case UnresolvedPropertyWithVarInjectionChecks:
    case Dynamic:
        return true;
    default:
        RELEASE_ASSERT_NOT_REACHED();
        return true;
    }
}

struct ResolveOp {
    ResolveOp(ResolveType type, size_t depth, Structure* structure, JSLexicalEnvironment* lexicalEnvironment, WatchpointSet* watchpointSet, uintptr_t operand, UniquedStringImpl* importedName = nullptr)
        : type(type)
        , depth(depth)
        , structure(structure)
        , lexicalEnvironment(lexicalEnvironment)
        , watchpointSet(watchpointSet)
        , operand(operand)
        , importedName(importedName)
    {
    }

    ResolveType type;
    size_t depth;
    Structure* structure;
    JSLexicalEnvironment* lexicalEnvironment;
    WatchpointSet* watchpointSet;
    uintptr_t operand;
    RefPtr<UniquedStringImpl> importedName;
};

class GetPutInfo {
    typedef unsigned Operand;
public:
    // Give each field 10 bits for simplicity.
    static_assert(sizeof(Operand) * 8 > 31, "Not enough bits for GetPutInfo");
    static constexpr unsigned isStrictShift = 30;
    static constexpr unsigned modeShift = 20;
    static constexpr unsigned initializationShift = 10;
    static constexpr unsigned typeBits = (1 << initializationShift) - 1;
    static constexpr unsigned initializationBits = ((1 << modeShift) - 1) & ~typeBits;
    static constexpr unsigned modeBits = ((1 << 30) - 1) & ~initializationBits & ~typeBits;
    static constexpr unsigned isStrictBit = 1 << 30;
    static_assert((modeBits & initializationBits & typeBits & isStrictBit) == 0x0, "There should be no intersection between ResolveMode ResolveType and InitializationMode");

    GetPutInfo() = default;

    GetPutInfo(ResolveMode resolveMode, ResolveType resolveType, InitializationMode initializationMode, ECMAMode ecmaMode)
        : m_operand((ecmaMode.isStrict() << isStrictShift) | (resolveMode << modeShift) | (static_cast<unsigned>(initializationMode) << initializationShift) | resolveType)
    {
    }

    explicit GetPutInfo(unsigned operand)
        : m_operand(operand)
    {
    }

    ResolveType resolveType() const { return static_cast<ResolveType>(m_operand & typeBits); }
    InitializationMode initializationMode() const { return static_cast<InitializationMode>((m_operand & initializationBits) >> initializationShift); }
    ResolveMode resolveMode() const { return static_cast<ResolveMode>((m_operand & modeBits) >> modeShift); }
    ECMAMode ecmaMode() const { return m_operand & isStrictBit ? ECMAMode::strict() : ECMAMode::sloppy(); }
    unsigned operand() const { return m_operand; }

    void dump(PrintStream&) const;

private:
    Operand m_operand { 0 };

    friend class JSC::LLIntOffsetsExtractor;
};

enum GetOrPut { Get, Put };

} // namespace JSC

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::ResolveMode);
void printInternal(PrintStream&, JSC::ResolveType);
void printInternal(PrintStream&, JSC::InitializationMode);

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
