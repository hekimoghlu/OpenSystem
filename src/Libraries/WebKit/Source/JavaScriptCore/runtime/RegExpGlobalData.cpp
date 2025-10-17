/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
#include "RegExpGlobalData.h"

#include "JSCInlines.h"

namespace JSC {

template<typename Visitor>
void RegExpGlobalData::visitAggregateImpl(Visitor& visitor)
{
    m_cachedResult.visitAggregate(visitor);
    m_substringGlobalAtomCache.visitAggregate(visitor);
}

DEFINE_VISIT_AGGREGATE(RegExpGlobalData);

JSValue RegExpGlobalData::getBackref(JSGlobalObject* globalObject, unsigned i)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSArray* array = m_cachedResult.lastResult(globalObject, globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    if (i < array->length()) {
        JSValue result = JSValue(array).get(globalObject, i);
        RETURN_IF_EXCEPTION(scope, { });
        ASSERT(result.isString() || result.isUndefined());
        if (!result.isUndefined())
            return result;
    }
    return jsEmptyString(vm);
}

JSValue RegExpGlobalData::getLastParen(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSArray* array = m_cachedResult.lastResult(globalObject, globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    unsigned length = array->length();
    if (length > 1) {
        JSValue result = JSValue(array).get(globalObject, length - 1);
        RETURN_IF_EXCEPTION(scope, { });
        ASSERT(result.isString() || result.isUndefined());
        if (!result.isUndefined())
            return result;
    }
    return jsEmptyString(vm);
}

JSValue RegExpGlobalData::getLeftContext(JSGlobalObject* globalObject)
{
    return m_cachedResult.leftContext(globalObject, globalObject);
}

JSValue RegExpGlobalData::getRightContext(JSGlobalObject* globalObject)
{
    return m_cachedResult.rightContext(globalObject, globalObject);
}

} // namespace JSC
