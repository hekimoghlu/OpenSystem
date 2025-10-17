/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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
#include "ClosureFileSystemNull.h"

using dyld3::closure::FileSystemNull;

bool FileSystemNull::getRealPath(const char possiblePath[MAXPATHLEN], char realPath[MAXPATHLEN]) const {
    return false;
}

bool FileSystemNull::loadFile(const char* path, LoadedFileInfo& info, char realerPath[MAXPATHLEN], void (^error)(const char* format, ...)) const {
    return false;
}

void FileSystemNull::unloadFile(const LoadedFileInfo& info) const {
    if (info.unload)
        info.unload(info);
}

void FileSystemNull::unloadPartialFile(LoadedFileInfo& info, uint64_t keepStartOffset, uint64_t keepLength) const {
    // Note we don't actually unload the data here, but we do want to update the offsets for other data structures to track where we are
    info.fileContent = (const void*)((char*)info.fileContent + keepStartOffset);
    info.fileContentLen = keepLength;
}

bool FileSystemNull::fileExists(const char* path, uint64_t* inode, uint64_t* mtime,
                                bool* issetuid, bool* inodesMatchRuntime) const {
    return false;
}
