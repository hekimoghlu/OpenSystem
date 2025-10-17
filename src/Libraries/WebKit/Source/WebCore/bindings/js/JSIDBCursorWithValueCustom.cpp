/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#include "JSIDBCursorWithValue.h"

#include "IDBBindingUtilities.h"
#include "IDBCursorWithValue.h"
#include <JavaScriptCore/JSCInlines.h>

namespace WebCore {
using namespace JSC;

JSC::JSValue JSIDBCursorWithValue::value(JSC::JSGlobalObject& lexicalGlobalObject) const
{
    auto throwScope = DECLARE_THROW_SCOPE(lexicalGlobalObject.vm());
    return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, wrapped().valueWrapper(), [&](JSC::ThrowScope&) {
        auto result = deserializeIDBValueWithKeyInjection(lexicalGlobalObject, wrapped().value(), wrapped().primaryKey(), wrapped().primaryKeyPath());
        return result ? result.value() : jsNull();
    });
}

template<typename Visitor>
void JSIDBCursorWithValue::visitAdditionalChildren(Visitor& visitor)
{
    JSIDBCursor::visitAdditionalChildren(visitor);
    wrapped().valueWrapper().visit(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSIDBCursorWithValue);

} // namespace WebCore
