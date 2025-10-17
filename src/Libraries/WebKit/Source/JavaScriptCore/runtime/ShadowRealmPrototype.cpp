/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#include "ShadowRealmPrototype.h"

#include "AuxiliaryBarrierInlines.h"
#include "IndirectEvalExecutable.h"
#include "Interpreter.h"
#include "JSGlobalObject.h"
#include "JSInternalPromise.h"
#include "JSModuleLoader.h"
#include "JSObjectInlines.h"
#include "ShadowRealmObject.h"
#include "StructureInlines.h"

#include "ShadowRealmPrototype.lut.h"

namespace JSC {

/* Source for ShadowRealmPrototype.lut.h
@begin shadowRealmPrototypeTable
  evaluate    JSBuiltin     DontEnum|Function  1
  importValue JSBuiltin     DontEnum|Function  2
@end
*/

const ClassInfo ShadowRealmPrototype::s_info = { "ShadowRealm"_s, &Base::s_info, &shadowRealmPrototypeTable, nullptr, CREATE_METHOD_TABLE(ShadowRealmPrototype) };

ShadowRealmPrototype::ShadowRealmPrototype(VM& vm, Structure* structure)
    : JSNonFinalObject(vm, structure)
{
}

void ShadowRealmPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

JSC_DEFINE_HOST_FUNCTION(importInRealm, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->uncheckedArgument(0);
    ShadowRealmObject* thisRealm = jsDynamicCast<ShadowRealmObject*>(thisValue);
    ASSERT(thisRealm);

    auto* promise = JSPromise::create(vm, globalObject->promiseStructure());

    auto sourceOrigin = callFrame->callerSourceOrigin(vm);
    auto* specifier = callFrame->uncheckedArgument(1).toString(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    JSGlobalObject* realmGlobalObject = thisRealm->globalObject();
    auto* internalPromise = realmGlobalObject->moduleLoader()->importModule(realmGlobalObject, specifier, jsUndefined(), sourceOrigin);
    RETURN_IF_EXCEPTION(scope, JSValue::encode(promise->rejectWithCaughtException(realmGlobalObject, scope)));

    scope.release();
    promise->resolve(globalObject, internalPromise);
    return JSValue::encode(promise);
}

JSC_DEFINE_HOST_FUNCTION(evalInRealm, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue thisValue = callFrame->argument(0);
    ShadowRealmObject* thisRealm = jsDynamicCast<ShadowRealmObject*>(thisValue);
    ASSERT(thisRealm);
    JSGlobalObject* realmGlobalObject = thisRealm->globalObject();

    JSValue evalArg = callFrame->argument(1);
    // eval code adapted from JSGlobalObjecFunctions::globalFuncEval
    auto script = asString(evalArg)->value(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    NakedPtr<JSObject> executableError;
    SourceTaintedOrigin sourceTaintedOrigin = computeNewSourceTaintedOriginFromStack(vm, callFrame);
    SourceCode source = makeSource(script, callFrame->callerSourceOrigin(vm), sourceTaintedOrigin);
    LexicallyScopedFeatures lexicallyScopedFeatures = globalObject->globalScopeExtension() ? TaintedByWithScopeLexicallyScopedFeature : NoLexicallyScopedFeatures;
    EvalExecutable* eval = IndirectEvalExecutable::create(realmGlobalObject, source, lexicallyScopedFeatures, DerivedContextType::None, false, EvalContextType::None, executableError);
    if (executableError) {
        JSValue error = executableError.get();
        ErrorInstance* errorInstance = jsDynamicCast<ErrorInstance*>(error);
        if (errorInstance != nullptr && errorInstance->errorType() == ErrorType::SyntaxError) {
            scope.clearException();
            const String syntaxErrorMessage = errorInstance->sanitizedMessageString(globalObject);
            RETURN_IF_EXCEPTION(scope, { });
            return throwVMError(globalObject, scope, createSyntaxError(globalObject, syntaxErrorMessage));
        }
        auto typeError = createTypeErrorCopy(globalObject, error);
        RETURN_IF_EXCEPTION(scope, { });
        return throwVMError(globalObject, scope, typeError);
    }
    RETURN_IF_EXCEPTION(scope, { });

    JSValue result = vm.interpreter.executeEval(eval, realmGlobalObject->globalThis(), realmGlobalObject->globalScope());
    if (UNLIKELY(scope.exception())) {
        NakedPtr<Exception> exception = scope.exception();
        JSValue error = exception->value();
        scope.clearException();
        auto typeError = createTypeErrorCopy(globalObject, error);
        RETURN_IF_EXCEPTION(scope, { });
        return throwVMError(globalObject, scope, typeError);
    }

    RELEASE_AND_RETURN(scope, JSValue::encode(result));
}

JSC_DEFINE_HOST_FUNCTION(moveFunctionToRealm, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue wrappedFnArg = callFrame->argument(0);
    JSFunction* wrappedFn = jsDynamicCast<JSFunction*>(wrappedFnArg);
    JSValue targetRealmArg = callFrame->argument(1);
    ShadowRealmObject* targetRealm = jsDynamicCast<ShadowRealmObject*>(targetRealmArg);
    ASSERT(targetRealm);
    RETURN_IF_EXCEPTION(scope, { });

    bool isBuiltin = false;
    JSGlobalObject* targetGlobalObj = targetRealm->globalObject();
    wrappedFn->setPrototype(vm, targetGlobalObj, targetGlobalObj->strictFunctionStructure(isBuiltin)->storedPrototype());
    RELEASE_AND_RETURN(scope, JSValue::encode(jsUndefined()));
}

} // namespace JSC
