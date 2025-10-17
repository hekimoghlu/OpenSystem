/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#include "IDBKeyRange.h"

#include "IDBBindingUtilities.h"
#include "IDBKey.h"
#include "IDBKeyData.h"
#include "ScriptExecutionContext.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <JavaScriptCore/JSGlobalObject.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBKeyRange);

Ref<IDBKeyRange> IDBKeyRange::create(RefPtr<IDBKey>&& lower, RefPtr<IDBKey>&& upper, bool isLowerOpen, bool isUpperOpen)
{
    return adoptRef(*new IDBKeyRange(WTFMove(lower), WTFMove(upper), isLowerOpen, isUpperOpen));
}

Ref<IDBKeyRange> IDBKeyRange::create(RefPtr<IDBKey>&& key)
{
    auto upper = key;
    return create(WTFMove(key), WTFMove(upper), false, false);
}

IDBKeyRange::IDBKeyRange(RefPtr<IDBKey>&& lower, RefPtr<IDBKey>&& upper, bool isLowerOpen, bool isUpperOpen)
    : m_lower(WTFMove(lower))
    , m_upper(WTFMove(upper))
    , m_isLowerOpen(isLowerOpen)
    , m_isUpperOpen(isUpperOpen)
{
}

IDBKeyRange::~IDBKeyRange() = default;

ExceptionOr<Ref<IDBKeyRange>> IDBKeyRange::only(RefPtr<IDBKey>&& key)
{
    if (!key || !key->isValid())
        return Exception { ExceptionCode::DataError };

    return create(WTFMove(key));
}

ExceptionOr<Ref<IDBKeyRange>> IDBKeyRange::only(JSGlobalObject& state, JSValue keyValue)
{
    VM& vm = state.vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    auto key = scriptValueToIDBKey(state, keyValue);
    EXCEPTION_ASSERT_UNUSED(scope, !scope.exception() || !key->isValid());
    return only(WTFMove(key));
}

ExceptionOr<Ref<IDBKeyRange>> IDBKeyRange::lowerBound(JSGlobalObject& state, JSValue boundValue, bool open)
{
    VM& vm = state.vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto bound = scriptValueToIDBKey(state, boundValue);
    EXCEPTION_ASSERT_UNUSED(scope, !scope.exception() || !bound->isValid());
    if (!bound->isValid())
        return Exception { ExceptionCode::DataError };

    return create(WTFMove(bound), nullptr, open, true);
}

ExceptionOr<Ref<IDBKeyRange>> IDBKeyRange::upperBound(JSGlobalObject& state, JSValue boundValue, bool open)
{
    VM& vm = state.vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto bound = scriptValueToIDBKey(state, boundValue);
    EXCEPTION_ASSERT_UNUSED(scope, !scope.exception() || !bound->isValid());
    if (!bound->isValid())
        return Exception { ExceptionCode::DataError };

    return create(nullptr, WTFMove(bound), true, open);
}

ExceptionOr<Ref<IDBKeyRange>> IDBKeyRange::bound(JSGlobalObject& state, JSValue lowerValue, JSValue upperValue, bool lowerOpen, bool upperOpen)
{
    VM& vm = state.vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto lower = scriptValueToIDBKey(state, lowerValue);
    EXCEPTION_ASSERT_UNUSED(scope, !scope.exception() || !lower->isValid());
    if (!lower->isValid())
        return Exception { ExceptionCode::DataError };
    auto upper = scriptValueToIDBKey(state, upperValue);
    EXCEPTION_ASSERT_UNUSED(scope, !scope.exception() || !upper->isValid());
    if (!upper->isValid())
        return Exception { ExceptionCode::DataError };
    if (upper->isLessThan(lower.get()))
        return Exception { ExceptionCode::DataError };
    if (upper->isEqual(lower.get()) && (lowerOpen || upperOpen))
        return Exception { ExceptionCode::DataError };

    return create(WTFMove(lower), WTFMove(upper), lowerOpen, upperOpen);
}

bool IDBKeyRange::isOnlyKey() const
{
    return m_lower && m_upper && !m_isLowerOpen && !m_isUpperOpen && m_lower->isEqual(*m_upper);
}

ExceptionOr<bool> IDBKeyRange::includes(JSC::JSGlobalObject& state, JSC::JSValue keyValue)
{
    VM& vm = state.vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto key = scriptValueToIDBKey(state, keyValue);
    EXCEPTION_ASSERT_UNUSED(scope, !scope.exception() || !key->isValid());
    if (!key->isValid())
        return Exception { ExceptionCode::DataError, "Failed to execute 'includes' on 'IDBKeyRange': The passed-in value is not a valid IndexedDB key."_s };

    if (m_lower) {
        int compare = m_lower->compare(key.get());

        if (compare > 0)
            return false;
        if (m_isLowerOpen && !compare)
            return false;
    }

    if (m_upper) {
        int compare = m_upper->compare(key.get());

        if (compare < 0)
            return false;
        if (m_isUpperOpen && !compare)
            return false;
    }

    return true;
}

} // namespace WebCore
