/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#include "FileSystemFileEntry.h"

#include "DOMException.h"
#include "DOMFileSystem.h"
#include "Document.h"
#include "ErrorCallback.h"
#include "File.h"
#include "FileCallback.h"
#include "WindowEventLoop.h"

namespace WebCore {

Ref<FileSystemFileEntry> FileSystemFileEntry::create(ScriptExecutionContext& context, DOMFileSystem& filesystem, const String& virtualPath)
{
    auto result = adoptRef(*new FileSystemFileEntry(context, filesystem, virtualPath));
    result->suspendIfNeeded();
    return result;
}

FileSystemFileEntry::FileSystemFileEntry(ScriptExecutionContext& context, DOMFileSystem& filesystem, const String& virtualPath)
    : FileSystemEntry(context, filesystem, virtualPath)
{
}

void FileSystemFileEntry::file(ScriptExecutionContext& context, Ref<FileCallback>&& successCallback, RefPtr<ErrorCallback>&& errorCallback)
{
    filesystem().getFile(context, *this, [this, pendingActivity = makePendingActivity(*this), successCallback = WTFMove(successCallback), errorCallback = WTFMove(errorCallback)](auto&& result) mutable {
        RefPtr document = this->document();
        if (!document)
            return;

        document->eventLoop().queueTask(TaskSource::Networking, [successCallback = WTFMove(successCallback), errorCallback = WTFMove(errorCallback), result = WTFMove(result), pendingActivity = WTFMove(pendingActivity)]() mutable {
            if (result.hasException()) {
                if (errorCallback)
                    errorCallback->handleEvent(DOMException::create(result.releaseException()));
                return;
            }
            successCallback->handleEvent(result.releaseReturnValue());
        });
    });
}

} // namespace WebCore
