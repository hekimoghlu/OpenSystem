/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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

namespace JSC {
    
enum class DerivedContextType : uint8_t { None, DerivedConstructorContext, DerivedMethodContext };
enum class EvalContextType    : uint8_t { None, FunctionEvalContext, InstanceFieldEvalContext };
enum class NeedsClassFieldInitializer : bool { No, Yes };

// FIXME: These flags, ParserModes and propagation to XXXCodeBlocks should be reorganized.
// https://bugs.webkit.org/show_bug.cgi?id=151547
struct ExecutableInfo {
    ExecutableInfo(bool isConstructor, PrivateBrandRequirement privateBrandRequirement, bool isBuiltinFunction, ConstructorKind constructorKind, JSParserScriptMode scriptMode, SuperBinding superBinding, SourceParseMode parseMode, DerivedContextType derivedContextType, NeedsClassFieldInitializer needsClassFieldInitializer, bool isArrowFunctionContext, bool isClassContext, EvalContextType evalContextType)
        : m_isConstructor(isConstructor)
        , m_privateBrandRequirement(static_cast<unsigned>(privateBrandRequirement))
        , m_isBuiltinFunction(isBuiltinFunction)
        , m_constructorKind(static_cast<unsigned>(constructorKind))
        , m_superBinding(static_cast<unsigned>(superBinding))
        , m_scriptMode(static_cast<unsigned>(scriptMode))
        , m_parseMode(parseMode)
        , m_derivedContextType(static_cast<unsigned>(derivedContextType))
        , m_needsClassFieldInitializer(static_cast<unsigned>(needsClassFieldInitializer))
        , m_isArrowFunctionContext(isArrowFunctionContext)
        , m_isClassContext(isClassContext)
        , m_evalContextType(static_cast<unsigned>(evalContextType))
    {
        ASSERT(m_constructorKind == static_cast<unsigned>(constructorKind));
        ASSERT(m_superBinding == static_cast<unsigned>(superBinding));
        ASSERT(m_scriptMode == static_cast<unsigned>(scriptMode));
    }

    bool isConstructor() const { return m_isConstructor; }
    PrivateBrandRequirement privateBrandRequirement() const { return static_cast<PrivateBrandRequirement>(m_privateBrandRequirement); }
    bool isBuiltinFunction() const { return m_isBuiltinFunction; }
    ConstructorKind constructorKind() const { return static_cast<ConstructorKind>(m_constructorKind); }
    SuperBinding superBinding() const { return static_cast<SuperBinding>(m_superBinding); }
    JSParserScriptMode scriptMode() const { return static_cast<JSParserScriptMode>(m_scriptMode); }
    SourceParseMode parseMode() const { return m_parseMode; }
    DerivedContextType derivedContextType() const { return static_cast<DerivedContextType>(m_derivedContextType); }
    EvalContextType evalContextType() const { return static_cast<EvalContextType>(m_evalContextType); }
    bool isArrowFunctionContext() const { return m_isArrowFunctionContext; }
    bool isClassContext() const { return m_isClassContext; }
    NeedsClassFieldInitializer needsClassFieldInitializer() const { return static_cast<NeedsClassFieldInitializer>(m_needsClassFieldInitializer); }

private:
    unsigned m_isConstructor : 1;
    unsigned m_privateBrandRequirement : 1;
    unsigned m_isBuiltinFunction : 1;
    unsigned m_constructorKind : 2;
    unsigned m_superBinding : 1;
    unsigned m_scriptMode: 1;
    SourceParseMode m_parseMode;
    unsigned m_derivedContextType : 2;
    unsigned m_needsClassFieldInitializer : 1;
    unsigned m_isArrowFunctionContext : 1;
    unsigned m_isClassContext : 1;
    unsigned m_evalContextType : 2;
};

} // namespace JSC
