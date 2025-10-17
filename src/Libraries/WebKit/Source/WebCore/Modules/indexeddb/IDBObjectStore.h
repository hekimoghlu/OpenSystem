/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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

#include "ActiveDOMObject.h"
#include "ExceptionOr.h"
#include "IDBCursorDirection.h"
#include "IDBIndexIdentifier.h"
#include "IDBKeyPath.h"
#include "IDBObjectStoreInfo.h"
#include <wtf/CheckedRef.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakRef.h>

namespace JSC {
class CallFrame;
class JSGlobalObject;
class JSValue;
class SlotVisitor;
}

namespace WebCore {

class DOMStringList;
class IDBIndex;
class IDBKey;
class IDBKeyRange;
class IDBRequest;
class IDBTransaction;
class SerializedScriptValue;
class WeakPtrImplWithEventTargetData;
class WebCoreOpaqueRoot;

struct IDBKeyRangeData;

namespace IndexedDB {
enum class ObjectStoreOverwriteMode : uint8_t;
}

class IDBObjectStore final : public ActiveDOMObject, public CanMakeCheckedPtr<IDBObjectStore> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBObjectStore);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(IDBObjectStore);
public:
    static UniqueRef<IDBObjectStore> create(ScriptExecutionContext&, const IDBObjectStoreInfo&, IDBTransaction&);
    ~IDBObjectStore();

    const String& name() const;
    ExceptionOr<void> setName(const String&);
    const std::optional<IDBKeyPath>& keyPath() const;
    Ref<DOMStringList> indexNames() const;
    IDBTransaction& transaction();
    bool autoIncrement() const;

    struct IndexParameters {
        bool unique;
        bool multiEntry;
    };

    ExceptionOr<Ref<IDBRequest>> openCursor(RefPtr<IDBKeyRange>&&, IDBCursorDirection);
    ExceptionOr<Ref<IDBRequest>> openCursor(JSC::JSGlobalObject&, JSC::JSValue key, IDBCursorDirection);
    ExceptionOr<Ref<IDBRequest>> openKeyCursor(RefPtr<IDBKeyRange>&&, IDBCursorDirection);
    ExceptionOr<Ref<IDBRequest>> openKeyCursor(JSC::JSGlobalObject&, JSC::JSValue key, IDBCursorDirection);
    ExceptionOr<Ref<IDBRequest>> get(JSC::JSGlobalObject&, JSC::JSValue key);
    ExceptionOr<Ref<IDBRequest>> get(IDBKeyRange*);
    ExceptionOr<Ref<IDBRequest>> getKey(JSC::JSGlobalObject&, JSC::JSValue key);
    ExceptionOr<Ref<IDBRequest>> getKey(IDBKeyRange*);
    ExceptionOr<Ref<IDBRequest>> add(JSC::JSGlobalObject&, JSC::JSValue, JSC::JSValue key);
    ExceptionOr<Ref<IDBRequest>> put(JSC::JSGlobalObject&, JSC::JSValue, JSC::JSValue key);
    ExceptionOr<Ref<IDBRequest>> deleteFunction(IDBKeyRange*);
    ExceptionOr<Ref<IDBRequest>> deleteFunction(JSC::JSGlobalObject&, JSC::JSValue key);
    ExceptionOr<Ref<IDBRequest>> clear();
    ExceptionOr<Ref<IDBIndex>> createIndex(const String& name, IDBKeyPath&&, const IndexParameters&);
    ExceptionOr<Ref<IDBIndex>> index(const String& name);
    ExceptionOr<void> deleteIndex(const String& name);
    ExceptionOr<Ref<IDBRequest>> count(IDBKeyRange*);
    ExceptionOr<Ref<IDBRequest>> count(JSC::JSGlobalObject&, JSC::JSValue key);
    ExceptionOr<Ref<IDBRequest>> getAll(RefPtr<IDBKeyRange>&&, std::optional<uint32_t> count);
    ExceptionOr<Ref<IDBRequest>> getAll(JSC::JSGlobalObject&, JSC::JSValue key, std::optional<uint32_t> count);
    ExceptionOr<Ref<IDBRequest>> getAllKeys(RefPtr<IDBKeyRange>&&, std::optional<uint32_t> count);
    ExceptionOr<Ref<IDBRequest>> getAllKeys(JSC::JSGlobalObject&, JSC::JSValue key, std::optional<uint32_t> count);

    ExceptionOr<Ref<IDBRequest>> putForCursorUpdate(JSC::JSGlobalObject&, JSC::JSValue, RefPtr<IDBKey>&&, RefPtr<SerializedScriptValue>&&);

    void markAsDeleted();
    bool isDeleted() const { return m_deleted; }

    const IDBObjectStoreInfo& info() const { return m_info; }

    void rollbackForVersionChangeAbort();

    // ActiveDOMObject.
    void ref() const final;
    void deref() const final;

    template<typename Visitor> void visitReferencedIndexes(Visitor&) const;
    void renameReferencedIndex(IDBIndex&, const String& newName);

private:
    IDBObjectStore(ScriptExecutionContext&, const IDBObjectStoreInfo&, IDBTransaction&);

    enum class InlineKeyCheck { Perform, DoNotPerform };
    ExceptionOr<Ref<IDBRequest>> putOrAdd(JSC::JSGlobalObject&, JSC::JSValue, RefPtr<IDBKey>, IndexedDB::ObjectStoreOverwriteMode, InlineKeyCheck, RefPtr<SerializedScriptValue>&& = nullptr);
    ExceptionOr<Ref<IDBRequest>> doCount(const IDBKeyRangeData&);
    ExceptionOr<Ref<IDBRequest>> doDelete(Function<ExceptionOr<RefPtr<IDBKeyRange>>()> &&);
    ExceptionOr<Ref<IDBRequest>> doOpenCursor(IDBCursorDirection, Function<ExceptionOr<RefPtr<IDBKeyRange>>()>&&);
    ExceptionOr<Ref<IDBRequest>> doOpenKeyCursor(IDBCursorDirection, Function<ExceptionOr<RefPtr<IDBKeyRange>>()>&&);
    ExceptionOr<Ref<IDBRequest>> doGetAll(std::optional<uint32_t> count, Function<ExceptionOr<RefPtr<IDBKeyRange>>()> &&);
    ExceptionOr<Ref<IDBRequest>> doGetAllKeys(std::optional<uint32_t> count, Function<ExceptionOr<RefPtr<IDBKeyRange>>()> &&);

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    IDBObjectStoreInfo m_info;
    IDBObjectStoreInfo m_originalInfo;

    // IDBObjectStore objects are always owned by their referencing IDBTransaction.
    // ObjectStores will never outlive transactions so its okay to keep a WeakRef here.
    WeakRef<IDBTransaction, WeakPtrImplWithEventTargetData> m_transaction;

    bool m_deleted { false };

    mutable Lock m_referencedIndexLock;
    HashMap<String, std::unique_ptr<IDBIndex>> m_referencedIndexes WTF_GUARDED_BY_LOCK(m_referencedIndexLock);
    HashMap<IDBIndexIdentifier, std::unique_ptr<IDBIndex>> m_deletedIndexes WTF_GUARDED_BY_LOCK(m_referencedIndexLock);
};

WebCoreOpaqueRoot root(IDBObjectStore*);

} // namespace WebCore
