/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#pragma once

#include "IDBCursor.h"
#include <wtf/TypeCasts.h>

namespace WebCore {

class IDBCursorWithValue final : public IDBCursor {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBCursorWithValue);
public:
    static Ref<IDBCursorWithValue> create(IDBObjectStore&, const IDBCursorInfo&);
    static Ref<IDBCursorWithValue> create(IDBIndex&, const IDBCursorInfo&);

    virtual ~IDBCursorWithValue();

    bool isKeyCursorWithValue() const  override { return true; }

private:
    IDBCursorWithValue(IDBObjectStore&, const IDBCursorInfo&);
    IDBCursorWithValue(IDBIndex&, const IDBCursorInfo&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::IDBCursorWithValue)
    static bool isType(const WebCore::IDBCursor& cursor) { return cursor.isKeyCursorWithValue(); }
SPECIALIZE_TYPE_TRAITS_END()
