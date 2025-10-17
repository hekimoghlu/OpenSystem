/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#include "FileSystemEntry.h"

#include "DOMException.h"
#include "DOMFileSystem.h"
#include "Document.h"
#include "ErrorCallback.h"
#include "FileSystemDirectoryEntry.h"
#include "FileSystemEntryCallback.h"
#include "ScriptExecutionContext.h"
#include "WindowEventLoop.h"
#include <wtf/FileSystem.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FileSystemEntry);

FileSystemEntry::FileSystemEntry(ScriptExecutionContext& context, DOMFileSystem& filesystem, const String& virtualPath)
    : ActiveDOMObject(&context)
    , m_filesystem(filesystem)
    , m_name(FileSystem::pathFileName(virtualPath))
    , m_virtualPath(virtualPath)
{
}

FileSystemEntry::~FileSystemEntry() = default;

DOMFileSystem& FileSystemEntry::filesystem() const
{
    return m_filesystem.get();
}

Document* FileSystemEntry::document() const
{
    return downcast<Document>(scriptExecutionContext());
}

void FileSystemEntry::getParent(ScriptExecutionContext& context, RefPtr<FileSystemEntryCallback>&& successCallback, RefPtr<ErrorCallback>&& errorCallback)
{
    if (!successCallback && !errorCallback)
        return;

    filesystem().getParent(context, *this, [this, pendingActivity = makePendingActivity(*this), successCallback = WTFMove(successCallback), errorCallback = WTFMove(errorCallback)]<typename Result> (Result&& result) mutable {
        RefPtr document = this->document();
        if (!document)
            return;

        document->eventLoop().queueTask(TaskSource::Networking, [successCallback = WTFMove(successCallback), errorCallback = WTFMove(errorCallback), result = std::forward<Result>(result), pendingActivity = WTFMove(pendingActivity)] () mutable {
            if (result.hasException()) {
                if (errorCallback)
                    errorCallback->handleEvent(DOMException::create(result.releaseException()));
                return;
            }
            if (successCallback)
                successCallback->handleEvent(result.releaseReturnValue());
        });
    });
}


} // namespace WebCore
