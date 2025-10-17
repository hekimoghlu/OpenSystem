/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#ifndef lf_hfs_dirpos_handler_h
#define lf_hfs_dirpos_handler_h

#include "lf_hfs_common.h"
#include "lf_hfs_catalog.h"

#define MAX_UTF8_NAME_LENGTH (NAME_MAX*3+1)

int LFHFS_MkDir         ( UVFSFileNode psDirNode, const char *pcName, const UVFSFileAttributes *psFileAttr, UVFSFileNode *ppsOutNode );
int LFHFS_RmDir         ( UVFSFileNode psDirNode, const char *pcUTF8Name, UVFSFileNode victimNode );
int LFHFS_Remove        ( UVFSFileNode psDirNode, const char *pcUTF8Name, UVFSFileNode victimNode);
int LFHFS_Lookup        ( UVFSFileNode psDirNode, const char *pcUTF8Name, UVFSFileNode *ppsOutNode );
int LFHFS_ReadDir       ( UVFSFileNode psDirNode, void* pvBuf, size_t iBufLen, uint64_t uCookie, size_t *iReadBytes, uint64_t *puVerifier );
int LFHFS_ReadDirAttr   ( UVFSFileNode psDirNode, void *pvBuf, size_t iBufLen, uint64_t uCookie, size_t *iReadBytes, uint64_t *puVerifier );
int LFHFS_ScanDir       ( UVFSFileNode psDirNode, scandir_matching_request_t* psMatchingCriteria, scandir_matching_reply_t* psMatchingResult );
int LFHFS_ScanIDs       ( UVFSFileNode psNode, __unused uint64_t uRequestedAttributes, const uint64_t* puFileIDArray, unsigned int iFileIDCount, scanids_match_block_t fMatchCallback);

int DIROPS_RemoveInternal( UVFSFileNode psDirNode, const char *pcUTF8Name, UVFSFileNode victimNode );
int DIROPS_LookupInternal( UVFSFileNode psDirNode, const char *pcUTF8Name, UVFSFileNode *ppsOutNode );
#endif /* lf_hfs_dirpos_handler_h */
