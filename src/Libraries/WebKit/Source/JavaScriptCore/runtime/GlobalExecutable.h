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

#include "ScriptExecutable.h"

namespace JSC {

class GlobalExecutable : public ScriptExecutable {
public:
    using Base = ScriptExecutable;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    DECLARE_INFO;

    unsigned lastLine() const { return m_lastLine; }
    unsigned endColumn() const { return m_endColumn; }

    void recordParse(CodeFeatures features, LexicallyScopedFeatures lexicallyScopedFeatures, bool hasCapturedVariables, int lastLine, unsigned endColumn)
    {
        Base::recordParse(features, lexicallyScopedFeatures, hasCapturedVariables);
        m_lastLine = lastLine;
        m_endColumn = endColumn;
        ASSERT(endColumn != UINT_MAX);
    }

    DECLARE_VISIT_CHILDREN;
    DECLARE_VISIT_OUTPUT_CONSTRAINTS;

    void finalizeUnconditionally(VM&, CollectionScope);

protected:
    friend class ScriptExecutable;
    GlobalExecutable(Structure* structure, VM& vm, const SourceCode& sourceCode, LexicallyScopedFeatures lexicallyScopedFeatures, DerivedContextType derivedContextType, bool isInArrowFunctionContext, bool isInsideOrdinaryFunction, EvalContextType evalContextType, Intrinsic intrinsic)
        : Base(structure, vm, sourceCode, lexicallyScopedFeatures, derivedContextType, isInArrowFunctionContext, isInsideOrdinaryFunction, evalContextType, intrinsic)
    {
    }

    CodeBlock* codeBlock() const
    {
        return m_codeBlock.get();
    }

    UnlinkedCodeBlock* unlinkedCodeBlock() const
    {
        return m_unlinkedCodeBlock.get();
    }

    CodeBlock* replaceCodeBlockWith(VM&, CodeBlock*);

    WriteBarrier<CodeBlock> m_codeBlock;
    WriteBarrier<UnlinkedCodeBlock> m_unlinkedCodeBlock;
    int m_lastLine { -1 };
    unsigned m_endColumn { UINT_MAX };
};

} // namespace JSC
