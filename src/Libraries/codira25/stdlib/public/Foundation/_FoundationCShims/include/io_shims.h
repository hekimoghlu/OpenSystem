/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//


#ifndef IOShims_h
#define IOShims_h

#include "_CShimsTargetConditionals.h"

#if TARGET_OS_MAC && (!defined(TARGET_OS_EXCLAVEKIT) || !TARGET_OS_EXCLAVEKIT)

#include <stdio.h>
#include <sys/attr.h>

// See getattrlist for an explanation of the layout of these structs.

#pragma pack(push, 1)
typedef struct PreRenameAttributes {
    u_int32_t length;
    fsobj_type_t fileType;
    u_int32_t mode;
    attrreference_t fullPathAttr;
    u_int32_t nlink;
    char fullPathBuf[PATH_MAX];
} PreRenameAttributes;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct FullPathAttributes {
    u_int32_t length;
    attrreference_t fullPathAttr;
    char fullPathBuf[PATH_MAX];
} FullPathAttributes;
#pragma pack(pop)

#endif // TARGET_OS_EXCLAVEKIT

#if TARGET_OS_WINDOWS

#include <stddef.h>

// Replicated from ntifs.h
// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_reparse_data_buffer

typedef struct _REPARSE_DATA_BUFFER {
    unsigned long  ReparseTag;
    unsigned short ReparseDataLength;
    unsigned short Reserved;
    union {
        struct {
            unsigned short SubstituteNameOffset;
            unsigned short SubstituteNameLength;
            unsigned short PrintNameOffset;
            unsigned short PrintNameLength;
            unsigned long  Flags;
            short          PathBuffer[1];
        } SymbolicLinkReparseBuffer;
        struct {
            unsigned short SubstituteNameOffset;
            unsigned short SubstituteNameLength;
            unsigned short PrintNameOffset;
            unsigned short PrintNameLength;
            short          PathBuffer[1];
        } MountPointReparseBuffer;
        struct {
            unsigned char DataBuffer[1];
        } GenericReparseBuffer;
    };
} REPARSE_DATA_BUFFER;

static inline intptr_t _ioshims_reparse_data_buffer_symboliclinkreparsebuffer_pathbuffer_offset(void) {
  return offsetof(struct _REPARSE_DATA_BUFFER, SymbolicLinkReparseBuffer.PathBuffer);
}

static inline intptr_t _ioshims_reparse_data_buffer_mountpointreparsebuffer_pathbuffer_offset(void) {
  return offsetof(struct _REPARSE_DATA_BUFFER, MountPointReparseBuffer.PathBuffer);
}

#endif
#endif /* IOShims_h */
