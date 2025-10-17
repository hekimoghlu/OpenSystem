/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#ifndef FileUtils_h
#define FileUtils_h

#include <stdint.h>

#include <map>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <TargetConditionals.h>
#include "Defines.h"
#if !TARGET_OS_EXCLAVEKIT
  #include <dispatch/dispatch.h>
#endif

#include "DyldSharedCache.h"
#include "JSON.h"

namespace cache_builder {
    struct FileAlias;
};

class Diagnostics;

//
// recursively walk all files in a directory tree
// symlinks are ignored
// dirFilter should return true on directories which should not be recursed into
// callback is called on each regular file found with stat() info about the file
//
void iterateDirectoryTree(const std::string& pathPrefix, const std::string& path, bool (^dirFilter)(const std::string& dirPath),
                          void (^callback)(const std::string& path, const struct stat& statBuf), bool processFiles=true, bool recurse=true);


//
// writes the buffer to a temp file, then renames the file to the final path
// returns true on success
//
bool safeSave(const void* buffer, size_t bufferLen, const std::string& path);


const void* mapFileReadOnly(const char* path, size_t& mappedSize);

bool fileExists(const std::string& path);

std::unordered_map<std::string, uint32_t> parseOrderFile(const std::string& orderFileData);
std::string loadOrderFile(const std::string& orderFilePath);

std::string normalize_absolute_file_path(std::string path);
std::string basePath(const std::string& path);
std::string dirPath(const std::string& path);
std::string realPath(const std::string& path);
std::string realFilePath(const std::string& path);

std::string toolDir();


#endif // FileUtils_h
