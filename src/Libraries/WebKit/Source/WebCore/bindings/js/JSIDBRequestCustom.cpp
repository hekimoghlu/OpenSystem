/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
#include "JSIDBRequest.h"

#include "IDBBindingUtilities.h"
#include "JSDOMConvertIndexedDB.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertSequences.h"
#include "JSIDBCursor.h"
#include "JSIDBDatabase.h"

namespace WebCore {
using namespace JSC;

JSC::JSValue JSIDBRequest::result(JSC::JSGlobalObject& lexicalGlobalObject) const
{
    auto throwScope = DECLARE_THROW_SCOPE(lexicalGlobalObject.vm());
    auto result = wrapped().result();
    if (UNLIKELY(result.hasException())) {
        propagateException(lexicalGlobalObject, throwScope, result.releaseException());
        return jsNull();
    }

    auto resultValue = result.releaseReturnValue();
    auto& resultWrapper = wrapped().resultWrapper();
    return WTF::switchOn(resultValue, [] (const IDBRequest::NullResultType& result) {
        if (result == IDBRequest::NullResultType::Empty)
            return JSC::jsNull();
        return JSC::jsUndefined();
    }, [] (uint64_t number) {
        return toJS<IDLUnsignedLongLong>(number);
    }, [&] (const RefPtr<IDBCursor>& cursor) {
        return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, resultWrapper, [&](JSC::ThrowScope& throwScope) {
            return toJS<IDLInterface<IDBCursor>>(lexicalGlobalObject, *jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject), throwScope, cursor.get());
        });
    }, [&] (const RefPtr<IDBDatabase>& database) {
        return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, resultWrapper, [&](JSC::ThrowScope& throwScope) {
            return toJS<IDLInterface<IDBDatabase>>(lexicalGlobalObject, *jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject), throwScope, database.get());
        });
    }, [&] (const IDBKeyData& keyData) {
        return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, resultWrapper, [&](JSC::ThrowScope&) {
            return toJS<IDLIDBKeyData>(lexicalGlobalObject, *jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject), keyData);
        });
    }, [&] (const Vector<IDBKeyData>& keyDatas) {
        return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, resultWrapper, [&](JSC::ThrowScope&) {
            return toJS<IDLSequence<IDLIDBKeyData>>(lexicalGlobalObject, *jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject), keyDatas);
        });
    }, [&] (const IDBGetResult& getResult) {
        return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, resultWrapper, [&](JSC::ThrowScope&) {
            auto result = deserializeIDBValueWithKeyInjection(lexicalGlobalObject, getResult.value(), getResult.keyData(), getResult.keyPath());
            return result ? result.value() : jsNull();
        });
    }, [&] (const IDBGetAllResult& getAllResult) {
        return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, resultWrapper, [&](JSC::ThrowScope& throwScope) {
            auto& keys = getAllResult.keys();
            auto& values = getAllResult.values();
            auto& keyPath = getAllResult.keyPath();
            JSC::MarkedArgumentBuffer list;
            list.ensureCapacity(values.size());
            for (unsigned i = 0; i < values.size(); i ++) {
                auto result = deserializeIDBValueWithKeyInjection(lexicalGlobalObject, values[i], keys[i], keyPath);
                if (!result)
                    return jsNull();
                list.append(result.value());
                if (UNLIKELY(list.hasOverflowed())) {
                    propagateException(lexicalGlobalObject, throwScope, Exception(ExceptionCode::UnknownError));
                    return jsNull();
                }
            }
            return JSValue(JSC::constructArray(&lexicalGlobalObject, static_cast<JSC::ArrayAllocationProfile*>(nullptr), list));
        });
    });
}

template<typename Visitor>
void JSIDBRequest::visitAdditionalChildren(Visitor& visitor)
{
    auto& request = wrapped();
    request.resultWrapper().visit(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSIDBRequest);

} // namespace WebCore
