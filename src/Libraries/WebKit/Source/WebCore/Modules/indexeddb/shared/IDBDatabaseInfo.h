/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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

#include "IDBIndexIdentifier.h"
#include "IDBObjectStoreIdentifier.h"
#include "IDBObjectStoreInfo.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class IDBDatabaseInfo {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(IDBDatabaseInfo, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT explicit IDBDatabaseInfo(const String& name, uint64_t version, uint64_t maxIndexID, HashMap<IDBObjectStoreIdentifier, IDBObjectStoreInfo>&& objectStoreMap = { });
    IDBDatabaseInfo() = default;

    enum IsolatedCopyTag { IsolatedCopy };
    IDBDatabaseInfo(const IDBDatabaseInfo&, IsolatedCopyTag);

    IDBDatabaseInfo isolatedCopy() const;

    const String& name() const { return m_name; }

    void setVersion(uint64_t version) { m_version = version; }
    uint64_t version() const { return m_version; }

    bool hasObjectStore(const String& name) const;
    IDBObjectStoreInfo createNewObjectStore(const String& name, std::optional<IDBKeyPath>&&, bool autoIncrement);
    void addExistingObjectStore(const IDBObjectStoreInfo&);
    IDBObjectStoreInfo* infoForExistingObjectStore(IDBObjectStoreIdentifier);
    IDBObjectStoreInfo* infoForExistingObjectStore(const String& objectStoreName);
    const IDBObjectStoreInfo* infoForExistingObjectStore(IDBObjectStoreIdentifier) const;
    const IDBObjectStoreInfo* infoForExistingObjectStore(const String& objectStoreName) const;

    void renameObjectStore(IDBObjectStoreIdentifier, const String& newName);

    Vector<String> objectStoreNames() const;
    const HashMap<IDBObjectStoreIdentifier, IDBObjectStoreInfo>& objectStoreMap() const { return m_objectStoreMap; }

    void deleteObjectStore(const String& objectStoreName);
    void deleteObjectStore(IDBObjectStoreIdentifier);

    void setMaxIndexID(uint64_t maxIndexID);
    IDBIndexIdentifier generateNextIndexID() { return IDBIndexIdentifier { ++m_maxIndexID }; }

#if !LOG_DISABLED
    String loggingString() const;
#endif

private:
    friend struct IPC::ArgumentCoder<IDBDatabaseInfo, void>;
    IDBObjectStoreInfo* getInfoForExistingObjectStore(const String& objectStoreName);
    IDBObjectStoreInfo* getInfoForExistingObjectStore(IDBObjectStoreIdentifier);

    String m_name;
    uint64_t m_version { 0 };
    uint64_t m_maxIndexID { 0 };

    HashMap<IDBObjectStoreIdentifier, IDBObjectStoreInfo> m_objectStoreMap;
};

} // namespace WebCore
