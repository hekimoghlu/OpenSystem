/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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

#include "FileSystemEntry.h"

namespace WebCore {

class ErrorCallback;
class FileSystemDirectoryReader;
class FileSystemEntryCallback;
class ScriptExecutionContext;

class FileSystemDirectoryEntry final : public FileSystemEntry {
public:
    static Ref<FileSystemDirectoryEntry> create(ScriptExecutionContext&, DOMFileSystem&, const String&);

    Ref<FileSystemDirectoryReader> createReader(ScriptExecutionContext&);

    struct Flags {
        bool create { false };
        bool exclusive { false };
    };

    void getFile(ScriptExecutionContext&, const String& path, const Flags& options, RefPtr<FileSystemEntryCallback>&&, RefPtr<ErrorCallback>&&);
    void getDirectory(ScriptExecutionContext&, const String& path, const Flags& options, RefPtr<FileSystemEntryCallback>&&, RefPtr<ErrorCallback>&&);

private:
    bool isDirectory() const final { return true; }
    using EntryMatchingFunction = Function<bool(const FileSystemEntry&)>;
    void getEntry(ScriptExecutionContext&, const String& path, const Flags& options, EntryMatchingFunction&&, RefPtr<FileSystemEntryCallback>&&, RefPtr<ErrorCallback>&&);

    FileSystemDirectoryEntry(ScriptExecutionContext&, DOMFileSystem&, const String& virtualPath);
};
static_assert(sizeof(FileSystemDirectoryEntry) == sizeof(FileSystemEntry));

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::FileSystemDirectoryEntry)
    static bool isType(const WebCore::FileSystemEntry& entry) { return entry.isDirectory(); }
SPECIALIZE_TYPE_TRAITS_END()
