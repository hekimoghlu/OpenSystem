/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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
#ifndef ClosureFileSystem_h
#define ClosureFileSystem_h

// For MAXPATHLEN
#include <sys/param.h>
// For va_list
#include <stdarg.h>
// For uint64_t
#include <stdint.h>

namespace dyld3 {
namespace closure {

struct LoadedFileInfo {
    const void*  fileContent                = nullptr;
    uint64_t     fileContentLen             = 0;
    uint64_t     sliceOffset                = 0;
    uint64_t     sliceLen            : 63,
                 isOSBinary          : 1;
    uint64_t     inode                      = 0;
    uint64_t     mtime                      = 0;
    void (*unload)(const LoadedFileInfo&)   = nullptr;
    const char*  path                       = nullptr;
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
class FileSystem {
protected:
    FileSystem() { }

public:

    // Get the real path for a given path, if it exists.
    // Returns true if the real path was found and updates the given buffer iff that is the case
    virtual bool getRealPath(const char possiblePath[MAXPATHLEN], char realPath[MAXPATHLEN]) const = 0;

    // Returns true on success.  If an error occurs the given callback will be called with the reason.
    // On success, info is filled with info about the loaded file.  If the path supplied includes a symlink,
    // the supplier realerPath is filled in with the real path of the file, otherwise it is set to the empty string.
    virtual bool loadFile(const char* path, LoadedFileInfo& info, char realerPath[MAXPATHLEN], void (^error)(const char* format, ...)) const = 0;

    // Frees the buffer allocated by loadFile()
    virtual void unloadFile(const LoadedFileInfo& info) const = 0;

    // Frees all but the requested range and adjusts info to new buffer location
    // Remaining buffer can be freed later with unloadFile()
    virtual void unloadPartialFile(LoadedFileInfo& info, uint64_t keepStartOffset, uint64_t keepLength) const = 0;

    // If a file exists at path, returns true and sets inode and mtime
    virtual bool fileExists(const char* path, uint64_t* inode=nullptr, uint64_t* mtime=nullptr,
                            bool* issetuid=nullptr, bool* inodesMatchRuntime = nullptr) const = 0;
};
#pragma clang diagnostic pop

} //  namespace closure
} //  namespace dyld3

#endif /* ClosureFileSystem_h */
