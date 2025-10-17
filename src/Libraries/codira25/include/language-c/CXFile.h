/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#ifndef LANGUAGE_CORE_C_CXFILE_H
#define LANGUAGE_CORE_C_CXFILE_H

#include <time.h>

#include "language/Core-c/CXString.h"
#include "language/Core-c/ExternC.h"
#include "language/Core-c/Platform.h"

LANGUAGE_CORE_C_EXTERN_C_BEGIN

/**
 * \defgroup CINDEX_FILES File manipulation routines
 *
 * @{
 */

/**
 * A particular source file that is part of a translation unit.
 */
typedef void *CXFile;

/**
 * Retrieve the complete file and path name of the given file.
 */
CINDEX_LINKAGE CXString clang_getFileName(CXFile SFile);

/**
 * Retrieve the last modification time of the given file.
 */
CINDEX_LINKAGE time_t clang_getFileTime(CXFile SFile);

/**
 * Uniquely identifies a CXFile, that refers to the same underlying file,
 * across an indexing session.
 */
typedef struct {
  unsigned long long data[3];
} CXFileUniqueID;

/**
 * Retrieve the unique ID for the given \c file.
 *
 * \param file the file to get the ID for.
 * \param outID stores the returned CXFileUniqueID.
 * \returns If there was a failure getting the unique ID, returns non-zero,
 * otherwise returns 0.
 */
CINDEX_LINKAGE int clang_getFileUniqueID(CXFile file, CXFileUniqueID *outID);

/**
 * Returns non-zero if the \c file1 and \c file2 point to the same file,
 * or they are both NULL.
 */
CINDEX_LINKAGE int clang_File_isEqual(CXFile file1, CXFile file2);

/**
 * Returns the real path name of \c file.
 *
 * An empty string may be returned. Use \c clang_getFileName() in that case.
 */
CINDEX_LINKAGE CXString clang_File_tryGetRealPathName(CXFile file);

/**
 * @}
 */

LANGUAGE_CORE_C_EXTERN_C_END

#endif
