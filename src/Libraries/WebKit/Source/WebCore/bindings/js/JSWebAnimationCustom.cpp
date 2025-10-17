/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#include "JSWebAnimation.h"

#include "Document.h"
#include "JSAnimationEffect.h"
#include "JSAnimationTimeline.h"
#include "JSCSSAnimation.h"
#include "JSCSSTransition.h"
#include "JSDOMConstructor.h"
#include "JSDOMConvert.h"

namespace WebCore {

using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<WebAnimation>&& value)
{
    if (value->isCSSAnimation())
        return createWrapper<CSSAnimation>(globalObject, WTFMove(value));
    if (value->isCSSTransition())
        return createWrapper<CSSTransition>(globalObject, WTFMove(value));
    return createWrapper<WebAnimation>(globalObject, WTFMove(value));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, WebAnimation& value)
{
    return wrap(lexicalGlobalObject, globalObject, value);
}

EncodedJSValue constructJSWebAnimation(JSGlobalObject* lexicalGlobalObject, CallFrame& callFrame)
{
    VM& vm = lexicalGlobalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    UNUSED_PARAM(throwScope);
    auto* jsConstructor = jsCast<JSDOMConstructorBase*>(callFrame.jsCallee());
    ASSERT(jsConstructor);
    auto* context = jsConstructor->scriptExecutionContext();
    if (UNLIKELY(!context))
        return throwConstructorScriptExecutionContextUnavailableError(*lexicalGlobalObject, throwScope, "Animation"_s);

    auto& document = downcast<Document>(*context);
    auto effect = convert<IDLNullable<IDLInterface<AnimationEffect>>>(*lexicalGlobalObject, callFrame.argument(0), [](JSGlobalObject& lexicalGlobalObject, ThrowScope& scope) {
        throwArgumentTypeError(lexicalGlobalObject, scope, 0, "effect"_s, "Animation"_s, nullptr, "AnimationEffect"_s);
    });
    if (UNLIKELY(effect.hasException(throwScope)))
        return encodedJSValue();

    if (callFrame.argument(1).isUndefined()) {
        auto object = WebAnimation::create(document, effect.releaseReturnValue());
        return JSValue::encode(toJSNewlyCreated<IDLInterface<WebAnimation>>(*lexicalGlobalObject, *jsConstructor->globalObject(), WTFMove(object)));
    }

    auto timeline = convert<IDLNullable<IDLInterface<AnimationTimeline>>>(*lexicalGlobalObject, callFrame.uncheckedArgument(1), [](JSGlobalObject& lexicalGlobalObject, ThrowScope& scope) {
        throwArgumentTypeError(lexicalGlobalObject, scope, 1, "timeline"_s, "Animation"_s, nullptr, "AnimationTimeline"_s);
    });
    if (UNLIKELY(timeline.hasException(throwScope)))
        return encodedJSValue();

    auto object = WebAnimation::create(document, effect.releaseReturnValue(), timeline.releaseReturnValue());
    return JSValue::encode(toJSNewlyCreated<IDLInterface<WebAnimation>>(*lexicalGlobalObject, *jsConstructor->globalObject(), WTFMove(object)));
}

} // namespace WebCore
