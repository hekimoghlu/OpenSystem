/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#ifndef ClosureFileSystemPhysical_h
#define ClosureFileSystemPhysical_h

#include "ClosureFileSystem.h"

#include <sys/stat.h>

namespace dyld3 {
namespace closure {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
class __attribute__((visibility("hidden"))) FileSystemPhysical : public FileSystem {
public:
    FileSystemPhysical(const char* rootPath=nullptr, const char* overlayPath=nullptr, bool allowRelativePaths=true)
            : FileSystem(), _rootPath(rootPath), _overlayPath(overlayPath), _allowRelativePaths(allowRelativePaths)
            { }

    bool getRealPath(const char possiblePath[MAXPATHLEN], char realPath[MAXPATHLEN]) const override;

    bool loadFile(const char* path, LoadedFileInfo& info, char realerPath[MAXPATHLEN], void (^error)(const char* format, ...)) const override;

    void unloadFile(const LoadedFileInfo& info) const override;

    void unloadPartialFile(LoadedFileInfo& info, uint64_t keepStartOffset, uint64_t keepLength) const override;

    bool fileExists(const char* path, uint64_t* inode=nullptr, uint64_t* mtime=nullptr,
                    bool* issetuid=nullptr, bool* inodesMatchRuntime = nullptr) const override;

private:

    void forEachPath(const char* path, void (^handler)(const char* fullPath, unsigned prefixLen, bool& stop)) const;

    const char* _rootPath;
    const char* _overlayPath;
    bool        _allowRelativePaths;
};
#pragma clang diagnostic pop

} //  namespace closure
} //  namespace dyld3

#endif /* ClosureFileSystemPhysical_h */
