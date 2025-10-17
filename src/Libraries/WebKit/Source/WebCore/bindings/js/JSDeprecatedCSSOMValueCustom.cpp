/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#include "JSDeprecatedCSSOMValue.h"

#include "JSCSSStyleDeclarationCustom.h"
#include "JSDeprecatedCSSOMPrimitiveValue.h"
#include "JSDeprecatedCSSOMValueList.h"
#include "JSNode.h"
#include "WebCoreOpaqueRootInlines.h"

namespace WebCore {
using namespace JSC;

bool JSDeprecatedCSSOMValueOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    JSDeprecatedCSSOMValue* jsCSSValue = jsCast<JSDeprecatedCSSOMValue*>(handle.slot()->asCell());
    if (!jsCSSValue->hasCustomProperties())
        return false;

    if (UNLIKELY(reason))
        *reason = "CSSStyleDeclaration is opaque root"_s;

    return containsWebCoreOpaqueRoot(visitor, jsCSSValue->wrapped().owner());
}

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<DeprecatedCSSOMValue>&& value)
{
    if (value->isValueList())
        return createWrapper<DeprecatedCSSOMValueList>(globalObject, WTFMove(value));
    // Expose CSS-wide keywords as plain CSSValues to keep the existing behavior.
    if (auto* primitiveValue = dynamicDowncast<DeprecatedCSSOMPrimitiveValue>(value.get()); primitiveValue && !primitiveValue->isCSSWideKeyword())
        return createWrapper<DeprecatedCSSOMPrimitiveValue>(globalObject, WTFMove(value));
    return createWrapper<DeprecatedCSSOMValue>(globalObject, WTFMove(value));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, DeprecatedCSSOMValue& value)
{
    return wrap(lexicalGlobalObject, globalObject, value);
}

} // namespace WebCore
