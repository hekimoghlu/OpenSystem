/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
#include "FileSystemDirectoryEntry.h"
#include "ScriptWrappable.h"
#include <wtf/RefCounted.h>
#include <wtf/WorkQueue.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class File;
class FileSystemFileEntry;
class FileSystemEntry;
class ScriptExecutionContext;

class DOMFileSystem final : public ScriptWrappable, public RefCounted<DOMFileSystem> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DOMFileSystem);
public:
    static Ref<FileSystemEntry> createEntryForFile(ScriptExecutionContext& context, Ref<File>&& file)
    {
        auto fileSystem = adoptRef(*new DOMFileSystem(WTFMove(file)));
        return fileSystem->fileAsEntry(context);
    }

    ~DOMFileSystem();

    const String& name() const { return m_name; }
    Ref<FileSystemDirectoryEntry> root(ScriptExecutionContext&);

    using DirectoryListingCallback = Function<void(ExceptionOr<Vector<Ref<FileSystemEntry>>>&&)>;
    void listDirectory(ScriptExecutionContext&, FileSystemDirectoryEntry&, DirectoryListingCallback&&);

    using GetParentCallback = Function<void(ExceptionOr<Ref<FileSystemDirectoryEntry>>&&)>;
    void getParent(ScriptExecutionContext&, FileSystemEntry&, GetParentCallback&&);

    using GetEntryCallback = Function<void(ExceptionOr<Ref<FileSystemEntry>>&&)>;
    void getEntry(ScriptExecutionContext&, FileSystemDirectoryEntry&, const String& virtualPath, const FileSystemDirectoryEntry::Flags&, GetEntryCallback&&);

    using GetFileCallback = Function<void(ExceptionOr<Ref<File>>&&)>;
    void getFile(ScriptExecutionContext&, FileSystemFileEntry&, GetFileCallback&&);

private:
    explicit DOMFileSystem(Ref<File>&&);

    String evaluatePath(StringView virtualPath);
    Ref<FileSystemEntry> fileAsEntry(ScriptExecutionContext&);

    String m_name;
    Ref<File> m_file;
    String m_rootPath;
    Ref<WorkQueue> m_workQueue;
};

} // namespace WebCore
