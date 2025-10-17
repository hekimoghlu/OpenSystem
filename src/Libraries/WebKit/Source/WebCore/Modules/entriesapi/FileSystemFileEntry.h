/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
class FileCallback;
class ScriptExecutionContext;

class FileSystemFileEntry final : public FileSystemEntry {
public:
    static Ref<FileSystemFileEntry> create(ScriptExecutionContext&, DOMFileSystem&, const String&);

    void file(ScriptExecutionContext&, Ref<FileCallback>&&, RefPtr<ErrorCallback>&& = nullptr);

private:
    bool isFile() const final { return true; }

    FileSystemFileEntry(ScriptExecutionContext&, DOMFileSystem&, const String& virtualPath);
};
static_assert(sizeof(FileSystemFileEntry) == sizeof(FileSystemEntry));

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::FileSystemFileEntry)
    static bool isType(const WebCore::FileSystemEntry& entry) { return entry.isFile(); }
SPECIALIZE_TYPE_TRAITS_END()
