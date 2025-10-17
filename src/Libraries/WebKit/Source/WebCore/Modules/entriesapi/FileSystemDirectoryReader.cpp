/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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
#include "FileSystemDirectoryReader.h"

#include "DOMException.h"
#include "DOMFileSystem.h"
#include "Document.h"
#include "ErrorCallback.h"
#include "FileSystemDirectoryEntry.h"
#include "FileSystemEntriesCallback.h"
#include "ScriptExecutionContext.h"
#include "WindowEventLoop.h"
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FileSystemDirectoryReader);

Ref<FileSystemDirectoryReader> FileSystemDirectoryReader::create(ScriptExecutionContext& context, FileSystemDirectoryEntry& directory)
{
    auto reader = adoptRef(*new FileSystemDirectoryReader(context, directory));
    reader->suspendIfNeeded();
    return reader;
}

FileSystemDirectoryReader::FileSystemDirectoryReader(ScriptExecutionContext& context, FileSystemDirectoryEntry& directory)
    : ActiveDOMObject(&context)
    , m_directory(directory)
{
}

FileSystemDirectoryReader::~FileSystemDirectoryReader() = default;

Document* FileSystemDirectoryReader::document() const
{
    return downcast<Document>(scriptExecutionContext());
}

// https://wicg.github.io/entries-api/#dom-filesystemdirectoryentry-readentries
void FileSystemDirectoryReader::readEntries(ScriptExecutionContext& context, Ref<FileSystemEntriesCallback>&& successCallback, RefPtr<ErrorCallback>&& errorCallback)
{
    if (m_isReading) {
        if (errorCallback)
            errorCallback->scheduleCallback(context, DOMException::create(Exception { ExceptionCode::InvalidStateError, "Directory reader is already reading"_s }));
        return;
    }

    if (m_error) {
        if (errorCallback)
            errorCallback->scheduleCallback(context, DOMException::create(*m_error));
        return;
    }

    if (m_isDone) {
        successCallback->scheduleCallback(context, { });
        return;
    }

    m_isReading = true;
    auto pendingActivity = makePendingActivity(*this);
    callOnMainThread([this, context = Ref { context }, successCallback = WTFMove(successCallback), errorCallback = WTFMove(errorCallback), pendingActivity = WTFMove(pendingActivity)]() mutable {
        m_isReading = false;
        m_directory->filesystem().listDirectory(context, m_directory, [this, successCallback = WTFMove(successCallback), errorCallback = WTFMove(errorCallback), pendingActivity = WTFMove(pendingActivity)](ExceptionOr<Vector<Ref<FileSystemEntry>>>&& result) mutable {
            RefPtr document = this->document();
            if (result.hasException()) {
                m_error = result.releaseException();
                if (errorCallback && document) {
                    document->eventLoop().queueTask(TaskSource::Networking, [this, errorCallback = WTFMove(errorCallback), pendingActivity = WTFMove(pendingActivity)]() mutable {
                        errorCallback->handleEvent(DOMException::create(*m_error));
                    });
                }
                return;
            }
            m_isDone = true;
            if (document) {
                document->eventLoop().queueTask(TaskSource::Networking, [successCallback = WTFMove(successCallback), pendingActivity = WTFMove(pendingActivity), result = result.releaseReturnValue()]() mutable {
                    successCallback->handleEvent(WTFMove(result));
                });
            }
        });
    });
}

} // namespace WebCore
