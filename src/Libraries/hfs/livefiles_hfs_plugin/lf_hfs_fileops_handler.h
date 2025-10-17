/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
#ifndef lf_hfs_fileops_handler_h
#define lf_hfs_fileops_handler_h

#include "lf_hfs_common.h"

#define VALID_IN_ATTR_MASK (    UVFS_FA_VALID_TYPE           |   \
                                UVFS_FA_VALID_MODE           |   \
                                UVFS_FA_VALID_NLINK          |   \
                                UVFS_FA_VALID_UID            |   \
                                UVFS_FA_VALID_GID            |   \
                                UVFS_FA_VALID_BSD_FLAGS      |   \
                                UVFS_FA_VALID_SIZE           |   \
                                UVFS_FA_VALID_ALLOCSIZE      |   \
                                UVFS_FA_VALID_FILEID         |   \
                                UVFS_FA_VALID_PARENTID       |   \
                                UVFS_FA_VALID_ATIME          |   \
                                UVFS_FA_VALID_MTIME          |   \
                                UVFS_FA_VALID_CTIME          |   \
                                UVFS_FA_VALID_BIRTHTIME)

#define VALID_OUT_ATTR_MASK (   UVFS_FA_VALID_TYPE           |   \
                                UVFS_FA_VALID_MODE           |   \
                                UVFS_FA_VALID_NLINK          |   \
                                UVFS_FA_VALID_UID            |   \
                                UVFS_FA_VALID_GID            |   \
                                UVFS_FA_VALID_BSD_FLAGS      |   \
                                UVFS_FA_VALID_SIZE           |   \
                                UVFS_FA_VALID_ALLOCSIZE      |   \
                                UVFS_FA_VALID_FILEID         |   \
                                UVFS_FA_VALID_PARENTID       |   \
                                UVFS_FA_VALID_ATIME          |   \
                                UVFS_FA_VALID_MTIME          |   \
                                UVFS_FA_VALID_CTIME          |   \
                                UVFS_FA_VALID_BIRTHTIME)

#define READ_ONLY_FA_FIELDS (   UVFS_FA_VALID_TYPE       |   \
                                UVFS_FA_VALID_NLINK      |   \
                                UVFS_FA_VALID_ALLOCSIZE  |   \
                                UVFS_FA_VALID_FILEID     |   \
                                UVFS_FA_VALID_PARENTID   |   \
                                UVFS_FA_VALID_CTIME      )

int LFHFS_Read        ( UVFSFileNode psNode, uint64_t uOffset, size_t iLength, void *pvBuf, size_t *iActuallyRead );
int LFHFS_Write       ( UVFSFileNode psNode, uint64_t uOffset, size_t iLength, const void *pvBuf, size_t *iActuallyWrite );
int LFHFS_Create      ( UVFSFileNode psNode, const char *pcName, const UVFSFileAttributes *psAttr, UVFSFileNode *ppsOutNode );
int LFHFS_GetAttr     ( UVFSFileNode psNode, UVFSFileAttributes *psOutAttr );
int LFHFS_SetAttr     ( UVFSFileNode psNode, const UVFSFileAttributes *psSetAttr, UVFSFileAttributes *psOutAttr );
int LFHFS_Reclaim     ( UVFSFileNode psNode, __unused unsigned int flags );
int LFHFS_ReadLink    ( UVFSFileNode psNode, void *pvOutBuf, size_t iBufSize, size_t *iActuallyRead, UVFSFileAttributes *psOutAttr );
int LFHFS_SymLink     ( UVFSFileNode psNode, const char *pcName, const char *psContent, const UVFSFileAttributes *psAttr, UVFSFileNode *ppsOutNode );
int LFHFS_Rename      ( UVFSFileNode psFromDirNode, UVFSFileNode psFromNode, const char *pcFromName, UVFSFileNode psToDirNode, UVFSFileNode psToNode, const char *pcToName, uint32_t flags);
int LFHFS_Link        ( UVFSFileNode psFromNode, UVFSFileNode psToDirNode, const char *pcToName, UVFSFileAttributes* psOutFileAttrs, UVFSFileAttributes* psOutDirAttrs );

int LFHFS_GetXAttr    ( UVFSFileNode psNode, const char *pcAttr, void *pvOutBuf, size_t iBufSize, size_t *iActualSize );
int LFHFS_SetXAttr    ( UVFSFileNode psNode, const char *pcAttr, const void *pvInBuf, size_t iBufSize, UVFSXattrHow How );
int LFHFS_ListXAttr   ( UVFSFileNode psNode, void *pvOutBuf, size_t iBufSize, size_t *iActualSize );

int LFHFS_StreamLookup ( UVFSFileNode psFileNode, UVFSStreamNode *ppsOutNode );
int LFHFS_StreamReclaim (UVFSStreamNode psStreamNode );
int LFHFS_StreamRead (UVFSStreamNode psStreamNode, uint64_t uOffset, size_t iLength, void *pvBuf, size_t *iActuallyRead );
#endif /* lf_hfs_fileops_handler_h */
