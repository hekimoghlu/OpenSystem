/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#include "JSKeyframeEffect.h"

#include "CSSPropertyNames.h"
#include "Document.h"
#include "JSDOMConvertObject.h"
#include "JSDOMConvertSequences.h"
#include "JSDOMConvertStrings.h"
#include "RenderStyle.h"

namespace WebCore {

using namespace JSC;

JSValue JSKeyframeEffect::getKeyframes(JSGlobalObject& lexicalGlobalObject, CallFrame&)
{
    auto lock = JSLockHolder { &lexicalGlobalObject };

    auto* context = jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject)->scriptExecutionContext();
    if (UNLIKELY(!context))
        return jsUndefined();

    auto& domGlobalObject = *jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject);
    auto computedKeyframes = wrapped().getKeyframes();
    auto keyframeObjects = computedKeyframes.map([&](auto& computedKeyframe) -> Strong<JSObject> {
        auto keyframeObject = convertDictionaryToJS(lexicalGlobalObject, domGlobalObject, { computedKeyframe });
        for (auto& [customProperty, propertyValue] : computedKeyframe.customStyleStrings) {
            auto value = toJS<IDLDOMString>(lexicalGlobalObject, propertyValue);
            JSObject::defineOwnProperty(keyframeObject, &lexicalGlobalObject, customProperty.impl(), PropertyDescriptor(value, 0), false);
        }
        for (auto& [propertyID, propertyValue] : computedKeyframe.styleStrings) {
            auto propertyName = KeyframeEffect::CSSPropertyIDToIDLAttributeName(propertyID);
            auto value = toJS<IDLDOMString>(lexicalGlobalObject, propertyValue);
            JSObject::defineOwnProperty(keyframeObject, &lexicalGlobalObject, AtomString(propertyName).impl(), PropertyDescriptor(value, 0), false);
        }
        return { lexicalGlobalObject.vm(), keyframeObject };
    });

    auto throwScope = DECLARE_THROW_SCOPE(lexicalGlobalObject.vm());
    return toJS<IDLSequence<IDLObject>>(lexicalGlobalObject, domGlobalObject, throwScope, keyframeObjects);
}

} // namespace WebCore
