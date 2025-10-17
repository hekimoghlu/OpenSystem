/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#include "FileSystemHandleIdentifier.h"
#include "IDLTypes.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

template<typename> class DOMPromiseDeferred;

class FileSystemStorageConnection;

class FileSystemHandle : public ActiveDOMObject, public RefCounted<FileSystemHandle> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FileSystemHandle);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    virtual ~FileSystemHandle();

    enum class Kind : uint8_t {
        File,
        Directory
    };
    Kind kind() const { return m_kind; }
    const String& name() const { return m_name; }
    FileSystemHandleIdentifier identifier() const { return m_identifier; }
    bool isClosed() const { return m_isClosed; }
    void close();

    void isSameEntry(FileSystemHandle&, DOMPromiseDeferred<IDLBoolean>&&) const;
    void move(FileSystemHandle&, const String& newName, DOMPromiseDeferred<void>&&);

protected:
    FileSystemHandle(ScriptExecutionContext&, Kind, String&& name, FileSystemHandleIdentifier, Ref<FileSystemStorageConnection>&&);
    FileSystemStorageConnection& connection() { return m_connection.get(); }

private:
    // ActiveDOMObject.
    void stop() final;

    Kind m_kind { Kind::File };
    String m_name;
    FileSystemHandleIdentifier m_identifier;
    Ref<FileSystemStorageConnection> m_connection;
    bool m_isClosed { false };
};

} // namespace WebCore
