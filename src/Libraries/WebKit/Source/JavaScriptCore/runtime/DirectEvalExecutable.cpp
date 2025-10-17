/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#include "config.h"
#include "DirectEvalExecutable.h"

#include "CodeCache.h"
#include "Debugger.h"
#include "Error.h"
#include "GlobalObjectMethodTable.h"
#include "JSCJSValueInlines.h"
#include "ParserError.h"

namespace JSC {

DirectEvalExecutable* DirectEvalExecutable::create(JSGlobalObject* globalObject, const SourceCode& source, LexicallyScopedFeatures lexicallyScopedFeatures, DerivedContextType derivedContextType, NeedsClassFieldInitializer needsClassFieldInitializer, PrivateBrandRequirement privateBrandRequirement, bool isArrowFunctionContext, bool isInsideOrdinaryFunction, EvalContextType evalContextType, const TDZEnvironment* variablesUnderTDZ, const PrivateNameEnvironment* privateNameEnvironment)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (!globalObject->evalEnabled()) {
        globalObject->globalObjectMethodTable()->reportViolationForUnsafeEval(globalObject, source.provider() ? source.provider()->source().toString() : nullString());
        throwException(globalObject, scope, createEvalError(globalObject, globalObject->evalDisabledErrorMessage()));
        return nullptr;
    }

    auto* executable = new (NotNull, allocateCell<DirectEvalExecutable>(vm)) DirectEvalExecutable(globalObject, source, lexicallyScopedFeatures, derivedContextType, needsClassFieldInitializer, privateBrandRequirement, isArrowFunctionContext, isInsideOrdinaryFunction, evalContextType);
    executable->finishCreation(vm);

    ParserError error;
    OptionSet<CodeGenerationMode> codeGenerationMode = globalObject->defaultCodeGenerationMode();

    // We don't bother with CodeCache here because direct eval uses a specialized DirectEvalCodeCache.
    UnlinkedEvalCodeBlock* unlinkedEvalCode = generateUnlinkedCodeBlockForDirectEval(vm, executable, executable->source(), JSParserScriptMode::Classic, codeGenerationMode, error, evalContextType, variablesUnderTDZ, privateNameEnvironment);

    if (globalObject->hasDebugger())
        globalObject->debugger()->sourceParsed(globalObject, executable->source().provider(), error.line(), error.message());

    if (error.isValid()) {
        throwVMError(globalObject, scope, error.toErrorObject(globalObject, executable->source()));
        return nullptr;
    }

    executable->m_unlinkedCodeBlock.set(vm, executable, unlinkedEvalCode);

    return executable;
}

DirectEvalExecutable::DirectEvalExecutable(JSGlobalObject* globalObject, const SourceCode& source, LexicallyScopedFeatures lexicallyScopedFeatures, DerivedContextType derivedContextType, NeedsClassFieldInitializer needsClassFieldInitializer, PrivateBrandRequirement privateBrandRequirement, bool isArrowFunctionContext, bool isInsideOrdinaryFunction, EvalContextType evalContextType)
    : EvalExecutable(globalObject, source, lexicallyScopedFeatures, derivedContextType, isArrowFunctionContext, isInsideOrdinaryFunction, evalContextType, needsClassFieldInitializer, privateBrandRequirement)
{
    ASSERT((needsClassFieldInitializer == NeedsClassFieldInitializer::No && privateBrandRequirement == PrivateBrandRequirement::None) || derivedContextType == DerivedContextType::DerivedConstructorContext);
}

} // namespace JSC
