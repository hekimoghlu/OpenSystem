/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#include "DirectoryFileListCreator.h"

#include "Document.h"
#include "FileChooser.h"
#include "FileList.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/FileSystem.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

DirectoryFileListCreator::~DirectoryFileListCreator()
{
    ASSERT(!m_completionHandler);
}

struct FileInformation {
    String path;
    String relativePath;
    String displayName;

    FileInformation isolatedCopy() const & { return { path.isolatedCopy(), relativePath.isolatedCopy(), displayName.isolatedCopy() }; }
    FileInformation isolatedCopy() && { return { WTFMove(path).isolatedCopy(), relativePath.isolatedCopy(), WTFMove(displayName).isolatedCopy() }; }
};

static void appendDirectoryFiles(const String& directory, const String& relativePath, Vector<FileInformation>& files)
{
    ASSERT(!isMainThread());
    for (auto& childName : FileSystem::listDirectory(directory)) {
        auto childPath = FileSystem::pathByAppendingComponent(directory, childName);
        if (FileSystem::isHiddenFile(childPath))
            continue;

        auto fileType = FileSystem::fileType(childPath);
        if (!fileType)
            continue;

        auto childRelativePath = makeString(relativePath, '/', childName);
        if (*fileType == FileSystem::FileType::Directory)
            appendDirectoryFiles(childPath, childRelativePath, files);
        else if (*fileType == FileSystem::FileType::Regular)
            files.append(FileInformation { childPath, childRelativePath, { } });
    }
}

static Vector<FileInformation> gatherFileInformation(const Vector<FileChooserFileInfo>& paths)
{
    ASSERT(!isMainThread());
    Vector<FileInformation> files;
    for (auto& info : paths) {
        if (FileSystem::fileType(info.path) == FileSystem::FileType::Directory)
            appendDirectoryFiles(info.path, FileSystem::pathFileName(info.path), files);
        else
            files.append(FileInformation { info.path, { }, info.displayName });
    }
    return files;
}

static Ref<FileList> toFileList(Document* document, const Vector<FileInformation>& files)
{
    ASSERT(isMainThread());
    auto fileObjects = files.map([document](auto& file) {
        if (file.relativePath.isNull())
            return File::create(document, file.path, { }, file.displayName);
        return File::createWithRelativePath(document, file.path, file.relativePath);
    });
    return FileList::create(WTFMove(fileObjects));
}

DirectoryFileListCreator::DirectoryFileListCreator(CompletionHandler&& completionHandler)
    : m_workQueue(WorkQueue::create("DirectoryFileListCreator Work Queue"_s))
    , m_completionHandler(WTFMove(completionHandler))
{
}

void DirectoryFileListCreator::start(Document* document, const Vector<FileChooserFileInfo>& paths)
{
    // Resolve directories on a background thread to avoid blocking the main thread.
    m_workQueue->dispatch([this, protectedThis = Ref { *this }, document = RefPtr { document }, paths = crossThreadCopy(paths)]() mutable {
        auto files = gatherFileInformation(paths);
        callOnMainThread([this, protectedThis = WTFMove(protectedThis), document = WTFMove(document), files = crossThreadCopy(files)]() mutable {
            if (auto completionHandler = std::exchange(m_completionHandler, nullptr))
                completionHandler(toFileList(document.get(), files));
        });
    });
}

void DirectoryFileListCreator::cancel()
{
    m_completionHandler = nullptr;
    m_workQueue = nullptr;
}

} // namespace WebCore
