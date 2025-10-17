/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#include "Exception.h"
#include "ScriptWrappable.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ErrorCallback;
class FileSystemDirectoryEntry;
class FileSystemEntriesCallback;
class ScriptExecutionContext;

class FileSystemDirectoryReader final : public ScriptWrappable, public ActiveDOMObject, public RefCounted<FileSystemDirectoryReader> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FileSystemDirectoryReader);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<FileSystemDirectoryReader> create(ScriptExecutionContext&, FileSystemDirectoryEntry&);
    ~FileSystemDirectoryReader();

    void readEntries(ScriptExecutionContext&, Ref<FileSystemEntriesCallback>&&, RefPtr<ErrorCallback>&&);

private:
    FileSystemDirectoryReader(ScriptExecutionContext&, FileSystemDirectoryEntry&);

    Document* document() const;

    Ref<FileSystemDirectoryEntry> m_directory;
    std::optional<Exception> m_error;
    bool m_isReading { false };
    bool m_isDone { false };
};

}
