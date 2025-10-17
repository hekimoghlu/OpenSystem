/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
#include "ParserTokens.h"
#include <wtf/Vector.h>
#include <wtf/text/UniquedStringImpl.h>
#include <wtf/text/WTFString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

struct SourceProviderCacheItemCreationParameters {
    unsigned lastTokenLine { 0 };
    unsigned lastTokenStartOffset { 0 };
    unsigned lastTokenEndOffset { 0 };
    unsigned lastTokenLineStartOffset { 0 };
    unsigned endFunctionOffset { 0 };
    unsigned parameterCount { 0 };
    LexicallyScopedFeatures lexicallyScopedFeatures { 0 };
    InnerArrowFunctionCodeFeatures innerArrowFunctionFeatures { 0 };
    Vector<UniquedStringImpl*, 8> usedVariables;
    JSTokenType tokenType { CLOSEBRACE };
    ConstructorKind constructorKind;
    SuperBinding expectedSuperBinding;
    bool needsFullActivation : 1 { false };
    bool usesEval : 1 { false };
    bool usesImportMeta : 1 { false };
    bool needsSuperBinding : 1 { false };
    bool isBodyArrowExpression : 1 { false };
};

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(SourceProviderCacheItem);
class SourceProviderCacheItem {
    WTF_MAKE_STRUCT_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(SourceProviderCacheItem);
public:
    static std::unique_ptr<SourceProviderCacheItem> create(const SourceProviderCacheItemCreationParameters&);
    ~SourceProviderCacheItem();

    JSToken endFunctionToken() const 
    {
        JSToken token;
        token.m_type = isBodyArrowExpression ? static_cast<JSTokenType>(tokenType) : CLOSEBRACE;
        token.m_data.offset = lastTokenStartOffset;
        token.m_location.startOffset = lastTokenStartOffset;
        token.m_location.endOffset = lastTokenEndOffset;
        token.m_location.line = lastTokenLine;
        token.m_location.lineStartOffset = lastTokenLineStartOffset;
        // token.m_location.sourceOffset is initialized once by the client. So,
        // we do not need to set it here.
        return token;
    }

    LexicallyScopedFeatures lexicallyScopedFeatures() const
    {
        LexicallyScopedFeatures features = NoLexicallyScopedFeatures;
        if (strictMode)
            features |= StrictModeLexicallyScopedFeature;
        if (taintedByWithScope)
            features |= TaintedByWithScopeLexicallyScopedFeature;
        return features;
    }

    bool needsFullActivation : 1;
    unsigned endFunctionOffset : 31;
    bool usesEval : 1;
    unsigned lastTokenLine : 31;
    bool strictMode : 1;
    unsigned lastTokenStartOffset : 31;
    unsigned expectedSuperBinding : 1; // SuperBinding
    unsigned lastTokenEndOffset: 31;
    bool needsSuperBinding: 1;
    unsigned parameterCount : 31;
    bool taintedByWithScope : 1;
    unsigned lastTokenLineStartOffset : 31;
    bool isBodyArrowExpression : 1;
    unsigned usedVariablesCount;
    unsigned tokenType : 24; // JSTokenType
    unsigned innerArrowFunctionFeatures : 6; // InnerArrowFunctionCodeFeatures
    unsigned constructorKind : 2; // ConstructorKind
    bool usesImportMeta : 1 { false };

    PackedPtr<UniquedStringImpl>* usedVariables() const { return const_cast<PackedPtr<UniquedStringImpl>*>(m_variables); }

private:
    SourceProviderCacheItem(const SourceProviderCacheItemCreationParameters&);

    PackedPtr<UniquedStringImpl> m_variables[0];
};

inline SourceProviderCacheItem::~SourceProviderCacheItem()
{
    for (unsigned i = 0; i < usedVariablesCount; ++i)
        m_variables[i]->deref();
}

inline std::unique_ptr<SourceProviderCacheItem> SourceProviderCacheItem::create(const SourceProviderCacheItemCreationParameters& parameters)
{
    size_t variableCount = parameters.usedVariables.size();
    size_t objectSize = sizeof(SourceProviderCacheItem) + sizeof(UniquedStringImpl*) * variableCount;
    void* slot = SourceProviderCacheItemMalloc::malloc(objectSize);
    return std::unique_ptr<SourceProviderCacheItem>(new (slot) SourceProviderCacheItem(parameters));
}

inline SourceProviderCacheItem::SourceProviderCacheItem(const SourceProviderCacheItemCreationParameters& parameters)
    : needsFullActivation(parameters.needsFullActivation)
    , endFunctionOffset(parameters.endFunctionOffset)
    , usesEval(parameters.usesEval)
    , lastTokenLine(parameters.lastTokenLine)
    , strictMode(parameters.lexicallyScopedFeatures & StrictModeLexicallyScopedFeature)
    , lastTokenStartOffset(parameters.lastTokenStartOffset)
    , expectedSuperBinding(static_cast<unsigned>(parameters.expectedSuperBinding))
    , lastTokenEndOffset(parameters.lastTokenEndOffset)
    , needsSuperBinding(parameters.needsSuperBinding)
    , parameterCount(parameters.parameterCount)
    , taintedByWithScope(parameters.lexicallyScopedFeatures & TaintedByWithScopeLexicallyScopedFeature)
    , lastTokenLineStartOffset(parameters.lastTokenLineStartOffset)
    , isBodyArrowExpression(parameters.isBodyArrowExpression)
    , usedVariablesCount(parameters.usedVariables.size())
    , tokenType(static_cast<unsigned>(parameters.tokenType))
    , innerArrowFunctionFeatures(static_cast<unsigned>(parameters.innerArrowFunctionFeatures))
    , constructorKind(static_cast<unsigned>(parameters.constructorKind))
    , usesImportMeta(parameters.usesImportMeta)
{
    ASSERT(tokenType == static_cast<unsigned>(parameters.tokenType));
    ASSERT(innerArrowFunctionFeatures == static_cast<unsigned>(parameters.innerArrowFunctionFeatures));
    ASSERT(constructorKind == static_cast<unsigned>(parameters.constructorKind));
    ASSERT(expectedSuperBinding == static_cast<unsigned>(parameters.expectedSuperBinding));
    for (unsigned i = 0; i < usedVariablesCount; ++i) {
        auto* pointer = parameters.usedVariables[i];
        pointer->ref();
        m_variables[i] = pointer;
    }
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
