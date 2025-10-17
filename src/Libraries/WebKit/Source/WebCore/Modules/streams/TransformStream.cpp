/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#include "TransformStream.h"

#include "JSDOMConvertObject.h"
#include "JSDOMConvertSequences.h"
#include "JSReadableStream.h"
#include "JSTransformStream.h"
#include "JSWritableStream.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/JSObjectInlines.h>

namespace WebCore {

struct CreateInternalTransformStreamResult {
    JSC::JSValue transform;
    Ref<ReadableStream> readable;
    Ref<WritableStream> writable;
};

static ExceptionOr<CreateInternalTransformStreamResult> createInternalTransformStream(JSDOMGlobalObject&, JSC::JSValue transformer, JSC::JSValue writableStrategy, JSC::JSValue readableStrategy);

ExceptionOr<Ref<TransformStream>> TransformStream::create(JSC::JSGlobalObject& globalObject, std::optional<JSC::Strong<JSC::JSObject>>&& transformer, std::optional<JSC::Strong<JSC::JSObject>>&& writableStrategy, std::optional<JSC::Strong<JSC::JSObject>>&& readableStrategy)
{
    JSC::JSValue transformerValue = JSC::jsUndefined();
    if (transformer)
        transformerValue = transformer->get();

    JSC::JSValue writableStrategyValue = JSC::jsUndefined();
    if (writableStrategy)
        writableStrategyValue = writableStrategy->get();

    JSC::JSValue readableStrategyValue = JSC::jsUndefined();
    if (readableStrategy)
        readableStrategyValue = readableStrategy->get();

    auto result = createInternalTransformStream(*JSC::jsCast<JSDOMGlobalObject*>(&globalObject), transformerValue, writableStrategyValue, readableStrategyValue);
    if (result.hasException())
        return result.releaseException();

    auto transformResult = result.releaseReturnValue();
    return adoptRef(*new TransformStream(transformResult.transform, WTFMove(transformResult.readable), WTFMove(transformResult.writable)));
}

TransformStream::TransformStream(JSC::JSValue internalTransformStream, Ref<ReadableStream>&& readable, Ref<WritableStream>&& writable)
    : m_internalTransformStream(internalTransformStream)
    , m_readable(WTFMove(readable))
    , m_writable(WTFMove(writable))
{
}

TransformStream::~TransformStream() = default;

static ExceptionOr<JSC::JSValue> invokeTransformStreamFunction(JSC::JSGlobalObject& globalObject, const JSC::Identifier& identifier, const JSC::MarkedArgumentBuffer& arguments)
{
    JSC::VM& vm = globalObject.vm();
    JSC::JSLockHolder lock(vm);

    auto scope = DECLARE_CATCH_SCOPE(vm);

    auto function = globalObject.get(&globalObject, identifier);
    ASSERT(function.isCallable());
    scope.assertNoExceptionExceptTermination();

    auto callData = JSC::getCallData(function);

    auto result = call(&globalObject, function, callData, JSC::jsUndefined(), arguments);
    RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });

    return result;
}

ExceptionOr<CreateInternalTransformStreamResult> createInternalTransformStream(JSDOMGlobalObject& globalObject, JSC::JSValue transformer, JSC::JSValue writableStrategy, JSC::JSValue readableStrategy)
{
    auto* clientData = static_cast<JSVMClientData*>(globalObject.vm().clientData);
    auto& privateName = clientData->builtinFunctions().transformStreamInternalsBuiltins().createInternalTransformStreamFromTransformerPrivateName();

    JSC::MarkedArgumentBuffer arguments;
    arguments.append(transformer);
    arguments.append(writableStrategy);
    arguments.append(readableStrategy);
    ASSERT(!arguments.hasOverflowed());

    auto result = invokeTransformStreamFunction(globalObject, privateName, arguments);
    if (UNLIKELY(result.hasException()))
        return result.releaseException();

    JSC::VM& vm = globalObject.vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto resultsConversionResult = convert<IDLSequence<IDLObject>>(globalObject, result.returnValue());
    if (UNLIKELY(resultsConversionResult.hasException(scope)))
        return Exception { ExceptionCode::ExistingExceptionError };

    auto results = resultsConversionResult.releaseReturnValue();
    ASSERT(results.size() == 3);

    return CreateInternalTransformStreamResult { results[0].get(), JSC::jsDynamicCast<JSReadableStream*>(results[1].get())->wrapped(), JSC::jsDynamicCast<JSWritableStream*>(results[2].get())->wrapped() };
}

template<typename Visitor>
void JSTransformStream::visitAdditionalChildren(Visitor& visitor)
{
    wrapped().internalTransformStream().visit(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSTransformStream);

} // namespace WebCore
