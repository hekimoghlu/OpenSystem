/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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
#include "JSCSSTransformComponent.h"

#include "JSCSSMatrixComponent.h"
#include "JSCSSPerspective.h"
#include "JSCSSRotate.h"
#include "JSCSSScale.h"
#include "JSCSSSkew.h"
#include "JSCSSSkewX.h"
#include "JSCSSSkewY.h"
#include "JSCSSTranslate.h"
#include "JSDOMWrapperCache.h"
#include <JavaScriptCore/JSCJSValueInlines.h>

namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<CSSTransformComponent>&& value)
{
    switch (value->getType()) {
    case CSSTransformType::MatrixComponent:
        return createWrapper<CSSMatrixComponent>(globalObject, WTFMove(value));
    case CSSTransformType::Perspective:
        return createWrapper<CSSPerspective>(globalObject, WTFMove(value));
    case CSSTransformType::Rotate:
        return createWrapper<CSSRotate>(globalObject, WTFMove(value));
    case CSSTransformType::Scale:
        return createWrapper<CSSScale>(globalObject, WTFMove(value));
    case CSSTransformType::Skew:
        return createWrapper<CSSSkew>(globalObject, WTFMove(value));
    case CSSTransformType::SkewX:
        return createWrapper<CSSSkewX>(globalObject, WTFMove(value));
    case CSSTransformType::SkewY:
        return createWrapper<CSSSkewY>(globalObject, WTFMove(value));
    case CSSTransformType::Translate:
        return createWrapper<CSSTranslate>(globalObject, WTFMove(value));
    }
    ASSERT_NOT_REACHED();
    return createWrapper<CSSTransformComponent>(globalObject, WTFMove(value));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, CSSTransformComponent& object)
{
    return wrap(lexicalGlobalObject, globalObject, object);
}

} // namespace WebCore
