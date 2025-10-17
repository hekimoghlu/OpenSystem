/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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
#include "IDBIndexInfo.h"
#include "IDBRequest.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace JSC {
class CallFrame;
}

namespace WebCore {

class IDBKeyRange;
class WebCoreOpaqueRoot;

struct IDBKeyRangeData;

class IDBIndex final : public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBIndex);
public:
    static UniqueRef<IDBIndex> create(ScriptExecutionContext&, const IDBIndexInfo&, IDBObjectStore&);

    virtual ~IDBIndex();

    const String& name() const;
    ExceptionOr<void> setName(const String&);
    IDBObjectStore& objectStore();
    const IDBKeyPath& keyPath() const;
    bool unique() const;
    bool multiEntry() const;

    void rollbackInfoForVersionChangeAbort();

    ExceptionOr<Ref<IDBRequest>> openCursor(RefPtr<IDBKeyRange>&&, IDBCursorDirection);
    ExceptionOr<Ref<IDBRequest>> openCursor(JSC::JSGlobalObject&, JSC::JSValue key, IDBCursorDirection);
    ExceptionOr<Ref<IDBRequest>> openKeyCursor(RefPtr<IDBKeyRange>&&, IDBCursorDirection);
    ExceptionOr<Ref<IDBRequest>> openKeyCursor(JSC::JSGlobalObject&, JSC::JSValue key, IDBCursorDirection);

    ExceptionOr<Ref<IDBRequest>> count(IDBKeyRange*);
    ExceptionOr<Ref<IDBRequest>> count(JSC::JSGlobalObject&, JSC::JSValue key);

    ExceptionOr<Ref<IDBRequest>> get(IDBKeyRange*);
    ExceptionOr<Ref<IDBRequest>> get(JSC::JSGlobalObject&, JSC::JSValue key);
    ExceptionOr<Ref<IDBRequest>> getKey(IDBKeyRange*);
    ExceptionOr<Ref<IDBRequest>> getKey(JSC::JSGlobalObject&, JSC::JSValue key);

    ExceptionOr<Ref<IDBRequest>> getAll(RefPtr<IDBKeyRange>&&, std::optional<uint32_t> count);
    ExceptionOr<Ref<IDBRequest>> getAll(JSC::JSGlobalObject&, JSC::JSValue key, std::optional<uint32_t> count);
    ExceptionOr<Ref<IDBRequest>> getAllKeys(RefPtr<IDBKeyRange>&&, std::optional<uint32_t> count);
    ExceptionOr<Ref<IDBRequest>> getAllKeys(JSC::JSGlobalObject&, JSC::JSValue key, std::optional<uint32_t> count);

    const IDBIndexInfo& info() const { return m_info; }

    void markAsDeleted();
    bool isDeleted() const { return m_deleted; }

    // ActiveDOMObject.
    void ref() const final;
    void deref() const final;

    WebCoreOpaqueRoot opaqueRoot();

private:
    IDBIndex(ScriptExecutionContext&, const IDBIndexInfo&, IDBObjectStore&);

    ExceptionOr<Ref<IDBRequest>> doCount(const IDBKeyRangeData&);
    ExceptionOr<Ref<IDBRequest>> doGet(ExceptionOr<IDBKeyRangeData>);
    ExceptionOr<Ref<IDBRequest>> doGetKey(ExceptionOr<IDBKeyRangeData>);
    ExceptionOr<Ref<IDBRequest>> doOpenCursor(IDBCursorDirection, Function<ExceptionOr<RefPtr<IDBKeyRange>>()> &&);
    ExceptionOr<Ref<IDBRequest>> doOpenKeyCursor(IDBCursorDirection, Function<ExceptionOr<RefPtr<IDBKeyRange>>()> &&);
    ExceptionOr<Ref<IDBRequest>> doGetAll(std::optional<uint32_t> count, Function<ExceptionOr<RefPtr<IDBKeyRange>>()> &&);
    ExceptionOr<Ref<IDBRequest>> doGetAllKeys(std::optional<uint32_t> count, Function<ExceptionOr<RefPtr<IDBKeyRange>>()> &&);

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    IDBIndexInfo m_info;
    IDBIndexInfo m_originalInfo;

    bool m_deleted { false };

    // IDBIndex objects are always owned by their referencing IDBObjectStore.
    // Indexes will never outlive ObjectStores so its okay to keep a raw C++ reference here.
    CheckedRef<IDBObjectStore> m_objectStore;
};

WebCoreOpaqueRoot root(IDBIndex*);

} // namespace WebCore
