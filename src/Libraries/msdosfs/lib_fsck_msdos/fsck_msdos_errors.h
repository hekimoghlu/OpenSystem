/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#ifndef _FSCK_MSDOS_ERRORS_H
#define _FSCK_MSDOS_ERRORS_H

enum fsck_msdos_errors {
    /*
     * Starting from 100 to not collide with POSIX error codes which may be
     * returned from pread/pwrite during fsck run (to keep our error codes unique).
     */
    fsckErrQuickCheckDirty = 200,   /* File system was found dirty in quick check. */
    fsckErrBootRegionInvalid,       /* Couldn't read the boot region, or it contains a non-recoverable corruption. */
    fsckErrCouldNotInitFAT,         /* Got a fatal error while initializing the FAT structure. */
    fsckErrCouldNotInitRootDir,     /* Got a fatal error while initializing the root dir structure. */
    fsckErrCouldNotScanDirs,        /* Got a fatal error while scanning the volume's directory hierarchy. */
    fsckErrCouldNotFreeUnused,      /* Got a fatal error while freeing unused clusters in the FAT. */
    fsckErrCannotRepairReadOnly,    /* Volume cannot be repaired because we're on read-only mode. */
    fsckErrCannotRepairAfterRetry,  /* Volume couldn't be repaired after one or more retries. */
};

#endif /* _FSCK_MSDOS_ERRORS_H */

