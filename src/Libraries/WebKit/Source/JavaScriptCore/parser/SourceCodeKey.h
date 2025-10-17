/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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

#include "ParserModes.h"
#include "UnlinkedSourceCode.h"
#include <wtf/HashTraits.h>

namespace JSC {

enum class SourceCodeType { EvalType, ProgramType, FunctionType, ModuleType };

class SourceCodeFlags {
    friend class CachedSourceCodeKey;

public:
    SourceCodeFlags() = default;

    SourceCodeFlags(
        SourceCodeType codeType, LexicallyScopedFeatures lexicallyScopedFeatures, JSParserScriptMode scriptMode,
        DerivedContextType derivedContextType, EvalContextType evalContextType, bool isArrowFunctionContext,
        OptionSet<CodeGenerationMode> codeGenerationMode)
        : m_flags(
            (static_cast<unsigned>(codeGenerationMode.toRaw()) << 6) |
            (static_cast<unsigned>(scriptMode) << 5) |
            (static_cast<unsigned>(isArrowFunctionContext) << 4) |
            (static_cast<unsigned>(evalContextType) << 3) |
            (static_cast<unsigned>(derivedContextType) << 2) |
            (static_cast<unsigned>(codeType) << 1) |
            (static_cast<unsigned>(lexicallyScopedFeatures & StrictModeLexicallyScopedFeature))
        )
    {
    }

    friend bool operator==(const SourceCodeFlags&, const SourceCodeFlags&) = default;

    unsigned bits() { return m_flags; }

private:
    unsigned m_flags { 0 };
};

class SourceCodeKey {
    friend class CachedSourceCodeKey;

public:
    SourceCodeKey()
    {
    }

    SourceCodeKey(
        const UnlinkedSourceCode& sourceCode, const String& name, SourceCodeType codeType, LexicallyScopedFeatures lexicallyScopedFeatures,
        JSParserScriptMode scriptMode, DerivedContextType derivedContextType, EvalContextType evalContextType, bool isArrowFunctionContext,
        OptionSet<CodeGenerationMode> codeGenerationMode, std::optional<int> functionConstructorParametersEndPosition)
            : m_sourceCode(sourceCode)
            , m_name(name)
            , m_flags(codeType, lexicallyScopedFeatures, scriptMode, derivedContextType, evalContextType, isArrowFunctionContext, codeGenerationMode)
            , m_functionConstructorParametersEndPosition(functionConstructorParametersEndPosition.value_or(-1))
            , m_hash(sourceCode.hash() ^ m_flags.bits())
    {
    }

    SourceCodeKey(WTF::HashTableDeletedValueType)
        : m_sourceCode(WTF::HashTableDeletedValue)
    {
    }

    bool isHashTableDeletedValue() const { return m_sourceCode.isHashTableDeletedValue(); }

    unsigned hash() const { return m_hash; }

    const UnlinkedSourceCode& source() const { return m_sourceCode; }

    size_t length() const { return m_sourceCode.length(); }

    bool isNull() const { return m_sourceCode.isNull(); }

    // To save memory, we compute our string on demand. It's expected that source
    // providers cache their strings to make this efficient.
    StringView string() const { return m_sourceCode.view(); }

    StringView host() const { return m_sourceCode.provider().sourceOrigin().url().host(); }

    bool operator==(const SourceCodeKey& other) const
    {
        return m_hash == other.m_hash
            && length() == other.length()
            && m_flags == other.m_flags
            && m_functionConstructorParametersEndPosition == other.m_functionConstructorParametersEndPosition
            && m_name == other.m_name
            && host() == other.host()
            && (m_sourceCode == other.m_sourceCode || string() == other.string());
    }

    struct Hash {
        static unsigned hash(const SourceCodeKey& key) { return key.hash(); }
        static bool equal(const SourceCodeKey& a, const SourceCodeKey& b) { return a == b; }
        static constexpr bool safeToCompareToEmptyOrDeleted = false;
    };

    struct HashTraits : SimpleClassHashTraits<SourceCodeKey> {
        static constexpr bool hasIsEmptyValueFunction = true;
        static bool isEmptyValue(const SourceCodeKey& key) { return key.isNull(); }
    };

private:
    UnlinkedSourceCode m_sourceCode;
    String m_name;
    SourceCodeFlags m_flags;
    int m_functionConstructorParametersEndPosition;
    unsigned m_hash;
};

} // namespace JSC
