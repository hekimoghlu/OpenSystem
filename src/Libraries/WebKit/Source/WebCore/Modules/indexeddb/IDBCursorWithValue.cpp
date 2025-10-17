/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#include "IDBCursorWithValue.h"

#include <JavaScriptCore/HeapInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBCursorWithValue);

Ref<IDBCursorWithValue> IDBCursorWithValue::create(IDBObjectStore& objectStore, const IDBCursorInfo& info)
{
    return adoptRef(*new IDBCursorWithValue(objectStore, info));
}

Ref<IDBCursorWithValue> IDBCursorWithValue::create(IDBIndex& index, const IDBCursorInfo& info)
{
    return adoptRef(*new IDBCursorWithValue(index, info));
}

IDBCursorWithValue::IDBCursorWithValue(IDBObjectStore& objectStore, const IDBCursorInfo& info)
    : IDBCursor(objectStore, info)
{
}

IDBCursorWithValue::IDBCursorWithValue(IDBIndex& index, const IDBCursorInfo& info)
    : IDBCursor(index, info)
{
}

IDBCursorWithValue::~IDBCursorWithValue() = default;

} // namespace WebCore
