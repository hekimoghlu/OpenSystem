/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
#include "PropertySlot.h"

#include "GetterSetter.h"
#include "JSCJSValueInlines.h"
#include "JSObject.h"

namespace JSC {

JSValue PropertySlot::functionGetter(JSGlobalObject* globalObject) const
{
    ASSERT(m_thisValue);
    return m_data.getter.getterSetter->callGetter(globalObject, m_thisValue);
}

JSValue PropertySlot::customGetter(VM& vm, PropertyName propertyName) const
{
    ASSERT(m_slotBase);
    JSGlobalObject* globalObject = m_slotBase->globalObject();
    JSValue thisValue = m_attributes & PropertyAttribute::CustomAccessor ? m_thisValue : JSValue(slotBase());
    if (auto domAttribute = this->domAttribute()) {
        if (!thisValue.inherits(domAttribute->classInfo)) {
            auto scope = DECLARE_THROW_SCOPE(vm);
            return throwDOMAttributeGetterTypeError(globalObject, scope, domAttribute->classInfo, propertyName);
        }
    }
    return JSValue::decode(m_data.custom.getValue(globalObject, JSValue::encode(thisValue), propertyName));
}

JSValue PropertySlot::getPureResult() const
{
    JSValue result;
    if (isTaintedByOpaqueObject())
        result = jsNull();
    else if (isCacheableValue())
        result = JSValue::decode(m_data.value);
    else if (isCacheableGetter())
        result = getterSetter();
    else if (isUnset())
        result = jsUndefined();
    else
        result = jsNull();
    
    return result;
}

} // namespace JSC
