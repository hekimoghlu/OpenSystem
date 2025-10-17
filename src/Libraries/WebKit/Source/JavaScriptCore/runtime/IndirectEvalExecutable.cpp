/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
#include "IndirectEvalExecutable.h"

#include "CodeCache.h"
#include "Debugger.h"
#include "Error.h"
#include "GlobalObjectMethodTable.h"
#include "JSCJSValueInlines.h"
#include "ParserError.h"

namespace JSC {

template<typename ErrorHandlerFunctor>
inline IndirectEvalExecutable* IndirectEvalExecutable::createImpl(JSGlobalObject* globalObject, const SourceCode& source, LexicallyScopedFeatures lexicallyScopedFeatures, DerivedContextType derivedContextType, bool isArrowFunctionContext, EvalContextType evalContextType, ErrorHandlerFunctor errorHandler)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (!globalObject->evalEnabled()) {
        globalObject->globalObjectMethodTable()->reportViolationForUnsafeEval(globalObject, source.provider() ? source.provider()->source().toString() : nullString());
        throwException(globalObject, scope, createEvalError(globalObject, globalObject->evalDisabledErrorMessage()));
        return nullptr;
    }

    auto* executable = new (NotNull, allocateCell<IndirectEvalExecutable>(vm)) IndirectEvalExecutable(globalObject, source, lexicallyScopedFeatures, derivedContextType, isArrowFunctionContext, evalContextType);
    executable->finishCreation(vm);

    ParserError error;
    OptionSet<CodeGenerationMode> codeGenerationMode = globalObject->defaultCodeGenerationMode();

    UnlinkedEvalCodeBlock* unlinkedEvalCode = vm.codeCache()->getUnlinkedEvalCodeBlock(
        vm, executable, executable->source(), codeGenerationMode, error, evalContextType);

    if (globalObject->hasDebugger())
        globalObject->debugger()->sourceParsed(globalObject, executable->source().provider(), error.line(), error.message());

    if (error.isValid()) {
        errorHandler(executable->source(), &error);
        scope.release();
        return nullptr;
    }

    executable->m_unlinkedCodeBlock.set(vm, executable, unlinkedEvalCode);

    return executable;
}

IndirectEvalExecutable* IndirectEvalExecutable::create(JSGlobalObject* globalObject, const SourceCode& source, LexicallyScopedFeatures lexicallyScopedFeatures, DerivedContextType derivedContextType, bool isArrowFunctionContext, EvalContextType evalContextType, NakedPtr<JSObject>& resultingError)
{
    auto handleError = [&](const SourceCode& source, ParserError* error) {
        resultingError = error->toErrorObject(globalObject, source);
    };
    return createImpl(globalObject, source, lexicallyScopedFeatures, derivedContextType, isArrowFunctionContext, evalContextType, handleError);
}

IndirectEvalExecutable* IndirectEvalExecutable::tryCreate(JSGlobalObject* globalObject, const SourceCode& source, LexicallyScopedFeatures lexicallyScopedFeatures, DerivedContextType derivedContextType, bool isArrowFunctionContext, EvalContextType evalContextType)
{
    VM& vm = globalObject->vm();
    auto handleError = [&](const SourceCode& source, ParserError* error) {
        auto scope = DECLARE_THROW_SCOPE(vm);
        throwVMError(globalObject, scope, error->toErrorObject(globalObject, source));
    };
    return createImpl(globalObject, source, lexicallyScopedFeatures, derivedContextType, isArrowFunctionContext, evalContextType, handleError);
}

constexpr bool insideOrdinaryFunction = false;
IndirectEvalExecutable::IndirectEvalExecutable(JSGlobalObject* globalObject, const SourceCode& source, LexicallyScopedFeatures lexicallyScopedFeatures, DerivedContextType derivedContextType, bool isArrowFunctionContext, EvalContextType evalContextType)
    : EvalExecutable(globalObject, source, lexicallyScopedFeatures, derivedContextType, isArrowFunctionContext, insideOrdinaryFunction, evalContextType, NeedsClassFieldInitializer::No, PrivateBrandRequirement::None)
{
}

} // namespace JSC
