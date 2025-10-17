/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
#include "PaintWorkletGlobalScope.h"

#include "Document.h"
#include "JSCSSPaintCallback.h"
#include "JSDOMConvert.h"
#include "LocalDOMWindow.h"
#include "RenderView.h"
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PaintWorkletGlobalScope);

RefPtr<PaintWorkletGlobalScope> PaintWorkletGlobalScope::tryCreate(Document& document, ScriptSourceCode&& code)
{
    RefPtr<VM> vm = VM::tryCreate();
    if (!vm)
        return nullptr;
    auto scope = adoptRef(*new PaintWorkletGlobalScope(document, vm.releaseNonNull(), WTFMove(code)));
    scope->addToContextsMap();
    return scope;
}

PaintWorkletGlobalScope::PaintWorkletGlobalScope(Document& document, Ref<VM>&& vm, ScriptSourceCode&& code)
    : WorkletGlobalScope(document, WTFMove(vm), WTFMove(code))
{
}

double PaintWorkletGlobalScope::devicePixelRatio() const
{
    if (!responsibleDocument() || !responsibleDocument()->domWindow())
        return 1.0;
    return responsibleDocument()->domWindow()->devicePixelRatio();
}

PaintDefinition::PaintDefinition(const AtomString& name, JSC::JSObject* paintConstructor, Ref<CSSPaintCallback>&& paintCallback, Vector<AtomString>&& inputProperties, Vector<String>&& inputArguments)
    : name(name)
    , paintConstructor(paintConstructor)
    , paintCallback(WTFMove(paintCallback))
    , inputProperties(WTFMove(inputProperties))
    , inputArguments(WTFMove(inputArguments))
{
}

// https://drafts.css-houdini.org/css-paint-api/#registering-custom-paint
ExceptionOr<void> PaintWorkletGlobalScope::registerPaint(JSC::JSGlobalObject& globalObject, const AtomString& name, Strong<JSObject> paintConstructor)
{
    auto& vm = paintConstructor->vm();
    JSC::JSLockHolder lock(vm);
    auto scope = DECLARE_THROW_SCOPE(vm);

    // Validate that paintConstructor is a VoidFunction
    if (!paintConstructor->isCallable())
        return Exception { ExceptionCode::TypeError, "paintConstructor must be callable"_s };

    if (name.isEmpty())
        return Exception { ExceptionCode::TypeError, "The first argument must not be the empty string"_s };

    {
        Locker locker { paintDefinitionLock() };

        if (paintDefinitionMap().contains(name))
            return Exception { ExceptionCode::InvalidModificationError, "This name has already been registered"_s };

        JSValue inputPropertiesIterableValue = paintConstructor->get(&globalObject, Identifier::fromString(vm, "inputProperties"_s));
        RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

        Vector<AtomString> inputProperties;
        if (!inputPropertiesIterableValue.isUndefined()) {
            auto inputPropertiesConversionResult = convert<IDLSequence<IDLAtomStringAdaptor<IDLDOMString>>>(globalObject, inputPropertiesIterableValue);
            if (UNLIKELY(inputPropertiesConversionResult.hasException(scope)))
                return Exception { ExceptionCode::ExistingExceptionError };
            inputProperties = inputPropertiesConversionResult.releaseReturnValue();
        }

        // FIXME: Validate input properties here (step 7).

        JSValue inputArgumentsIterableValue = paintConstructor->get(&globalObject, Identifier::fromString(vm, "inputArguments"_s));
        RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

        Vector<String> inputArguments;
        if (!inputArgumentsIterableValue.isUndefined()) {
            auto inputArgumentsConversionResult = convert<IDLSequence<IDLDOMString>>(globalObject, inputArgumentsIterableValue);
            if (UNLIKELY(inputArgumentsConversionResult.hasException(scope)))
                return Exception { ExceptionCode::ExistingExceptionError };
            inputArguments = inputArgumentsConversionResult.releaseReturnValue();
        }

        // FIXME: Parse syntax for inputArguments here (steps 11 and 12).

        JSValue contextOptionsValue = paintConstructor->get(&globalObject, Identifier::fromString(vm, "contextOptions"_s));
        RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });
        UNUSED_PARAM(contextOptionsValue);

        // FIXME: Convert to PaintRenderingContext2DSettings here (step 14).

        if (!paintConstructor->isConstructor())
            return Exception { ExceptionCode::TypeError, "The second argument must be a constructor"_s };

        JSValue prototypeValue = paintConstructor->get(&globalObject, vm.propertyNames->prototype);
        RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

        if (!prototypeValue.isObject())
            return Exception { ExceptionCode::TypeError, "The second argument must have a prototype that is an object"_s };

        JSValue paintValue = prototypeValue.get(&globalObject, Identifier::fromString(vm, "paint"_s));
        RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

        if (paintValue.isUndefined())
            return Exception { ExceptionCode::TypeError, "The class must have a paint method"_s };

        auto paintCallback = convert<IDLCallbackFunction<JSCSSPaintCallback>>(globalObject, paintValue, *jsCast<JSDOMGlobalObject*>(&globalObject));
        if (UNLIKELY(paintCallback.hasException(scope)))
            return Exception { ExceptionCode::ExistingExceptionError };

        auto paintDefinition = makeUnique<PaintDefinition>(name, paintConstructor.get(), paintCallback.releaseReturnValue(), WTFMove(inputProperties), WTFMove(inputArguments));
        paintDefinitionMap().add(name, WTFMove(paintDefinition));
    }

    // This is for the case when we have already visited the paint definition map, and the GC is currently running in the background.
    vm.writeBarrier(&globalObject);

    // FIXME: construct documentDefinition (step 22).

    // FIXME: we should only repaint affected custom paint images <https://bugs.webkit.org/show_bug.cgi?id=192322>.
    if (responsibleDocument() && responsibleDocument()->renderView())
        responsibleDocument()->renderView()->repaintRootContents();

    return { };
}

} // namespace WebCore
