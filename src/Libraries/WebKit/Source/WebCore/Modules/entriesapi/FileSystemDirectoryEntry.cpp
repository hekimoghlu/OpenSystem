/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#include "config.h"
#include "FileSystemDirectoryEntry.h"

#include "DOMException.h"
#include "DOMFileSystem.h"
#include "Document.h"
#include "ErrorCallback.h"
#include "FileSystemDirectoryReader.h"
#include "FileSystemEntryCallback.h"
#include "FileSystemFileEntry.h"
#include "ScriptExecutionContext.h"
#include "WindowEventLoop.h"

namespace WebCore {

Ref<FileSystemDirectoryEntry> FileSystemDirectoryEntry::create(ScriptExecutionContext& context, DOMFileSystem& filesystem, const String& virtualPath)
{
    auto result = adoptRef(*new FileSystemDirectoryEntry(context, filesystem, virtualPath));
    result->suspendIfNeeded();
    return result;
}

FileSystemDirectoryEntry::FileSystemDirectoryEntry(ScriptExecutionContext& context, DOMFileSystem& filesystem, const String& virtualPath)
    : FileSystemEntry(context, filesystem, virtualPath)
{
}

Ref<FileSystemDirectoryReader> FileSystemDirectoryEntry::createReader(ScriptExecutionContext& context)
{
    return FileSystemDirectoryReader::create(context, *this);
}

void FileSystemDirectoryEntry::getEntry(ScriptExecutionContext& context, const String& path, const Flags& flags, EntryMatchingFunction&& matches, RefPtr<FileSystemEntryCallback>&& successCallback, RefPtr<ErrorCallback>&& errorCallback)
{
    if (!successCallback && !errorCallback)
        return;

    filesystem().getEntry(context, *this, path, flags, [this, pendingActivity = makePendingActivity(*this), matches = WTFMove(matches), successCallback = WTFMove(successCallback), errorCallback = WTFMove(errorCallback)](auto&& result) mutable {
        RefPtr document = this->document();
        if (result.hasException()) {
            if (errorCallback && document) {
                document->eventLoop().queueTask(TaskSource::Networking, [errorCallback = WTFMove(errorCallback), exception = result.releaseException(), pendingActivity = WTFMove(pendingActivity)]() mutable {
                    errorCallback->handleEvent(DOMException::create(WTFMove(exception)));
                });
            }
            return;
        }
        auto entry = result.releaseReturnValue();
        if (!matches(entry)) {
            if (errorCallback && document) {
                document->eventLoop().queueTask(TaskSource::Networking, [errorCallback = WTFMove(errorCallback), pendingActivity = WTFMove(pendingActivity)]() mutable {
                    errorCallback->handleEvent(DOMException::create(Exception { ExceptionCode::TypeMismatchError, "Entry at given path does not match expected type"_s }));
                });
            }
            return;
        }
        if (successCallback && document) {
            document->eventLoop().queueTask(TaskSource::Networking, [successCallback = WTFMove(successCallback), entry = WTFMove(entry), pendingActivity = WTFMove(pendingActivity)]() mutable {
                successCallback->handleEvent(WTFMove(entry));
            });
        }
    });
}

void FileSystemDirectoryEntry::getFile(ScriptExecutionContext& context, const String& path, const Flags& flags, RefPtr<FileSystemEntryCallback>&& successCallback, RefPtr<ErrorCallback>&& errorCallback)
{
    getEntry(context, path, flags, [](auto& entry) { return entry.isFile(); }, WTFMove(successCallback), WTFMove(errorCallback));
}

void FileSystemDirectoryEntry::getDirectory(ScriptExecutionContext& context, const String& path, const Flags& flags, RefPtr<FileSystemEntryCallback>&& successCallback, RefPtr<ErrorCallback>&& errorCallback)
{
    getEntry(context, path, flags, [](auto& entry) { return entry.isDirectory(); }, WTFMove(successCallback), WTFMove(errorCallback));
}

} // namespace WebCore
