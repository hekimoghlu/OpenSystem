/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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

#include "ExceptionOr.h"
#include "IDBCursorDirection.h"
#include "IDBCursorInfo.h"
#include "IDBKeyPath.h"
#include "IDBRequest.h"
#include "IDBValue.h"
#include "JSValueInWrappedObject.h"
#include <JavaScriptCore/Strong.h>
#include <variant>
#include <wtf/WeakPtr.h>

namespace WebCore {

class IDBGetResult;
class IDBIndex;
class IDBObjectStore;
class IDBTransaction;

class IDBCursor : public ScriptWrappable, public RefCounted<IDBCursor> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBCursor);
public:
    static Ref<IDBCursor> create(IDBObjectStore&, const IDBCursorInfo&);
    static Ref<IDBCursor> create(IDBIndex&, const IDBCursorInfo&);
    
    virtual ~IDBCursor();

    using Source = std::variant<RefPtr<IDBObjectStore>, RefPtr<IDBIndex>>;

    const Source& source() const;
    IDBCursorDirection direction() const;

    IDBKey* key() { return m_key.get(); };
    IDBKey* primaryKey() { return m_primaryKey.get(); };
    IDBValue value() { return m_value; };
    const std::optional<IDBKeyPath>& primaryKeyPath() { return m_keyPath; };
    JSValueInWrappedObject& keyWrapper() { return m_keyWrapper; }
    JSValueInWrappedObject& primaryKeyWrapper() { return m_primaryKeyWrapper; }
    JSValueInWrappedObject& valueWrapper() { return m_valueWrapper; }

    ExceptionOr<Ref<IDBRequest>> update(JSC::JSGlobalObject&, JSC::JSValue);
    ExceptionOr<void> advance(unsigned);
    ExceptionOr<void> continueFunction(JSC::JSGlobalObject&, JSC::JSValue key);
    ExceptionOr<void> continuePrimaryKey(JSC::JSGlobalObject&, JSC::JSValue key, JSC::JSValue primaryKey);
    ExceptionOr<Ref<IDBRequest>> deleteFunction();

    ExceptionOr<void> continueFunction(const IDBKeyData&);

    const IDBCursorInfo& info() const { return m_info; }

    void setRequest(IDBRequest& request) { m_request = request; }
    void clearRequest() { m_request.clear(); }
    void clearWrappers();
    IDBRequest* request() { return m_request.get(); }

    bool setGetResult(IDBRequest&, const IDBGetResult&, uint64_t operationID);

    virtual bool isKeyCursorWithValue() const { return false; }

    std::optional<IDBGetResult> iterateWithPrefetchedRecords(unsigned count, uint64_t lastWriteOperationID);
    void clearPrefetchedRecords();

protected:
    IDBCursor(IDBObjectStore&, const IDBCursorInfo&);
    IDBCursor(IDBIndex&, const IDBCursorInfo&);

private:
    bool sourcesDeleted() const;
    IDBObjectStore& effectiveObjectStore() const;
    IDBTransaction& transaction() const;

    void uncheckedIterateCursor(const IDBKeyData&, unsigned count);
    void uncheckedIterateCursor(const IDBKeyData&, const IDBKeyData&);

    IDBCursorInfo m_info;
    Source m_source;
    WeakPtr<IDBRequest, WeakPtrImplWithEventTargetData> m_request;

    bool m_gotValue { false };

    RefPtr<IDBKey> m_key;
    RefPtr<IDBKey> m_primaryKey;
    IDBKeyData m_keyData;
    IDBKeyData m_primaryKeyData;
    IDBValue m_value;
    std::optional<IDBKeyPath> m_keyPath;

    JSValueInWrappedObject m_keyWrapper;
    JSValueInWrappedObject m_primaryKeyWrapper;
    JSValueInWrappedObject m_valueWrapper;

    Deque<IDBCursorRecord> m_prefetchedRecords;
    uint64_t m_prefetchOperationID { 0 };
};


inline const IDBCursor::Source& IDBCursor::source() const
{
    return m_source;
}

inline IDBCursorDirection IDBCursor::direction() const
{
    return m_info.cursorDirection();
}

} // namespace WebCore
