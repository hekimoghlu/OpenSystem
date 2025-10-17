/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#ifndef _KERNELCACHE_H_
#define _KERNELCACHE_H_

#include <libc.h>
#include "kext_tools_util.h"

#define PLATFORM_NAME_LEN  (64)
#define ROOT_PATH_LEN     (256)

#define COMP_TYPE_LZSS      'lzss'
#define COMP_TYPE_FASTLIB   'lzvn'


// prelinkVersion value >= 1 means KASLR supported
typedef struct prelinked_kernel_header {
    uint32_t  signature;
    uint32_t  compressType;
    uint32_t  adler32;
    uint32_t  uncompressedSize;
    uint32_t  compressedSize;
    uint32_t  prelinkVersion;
    uint32_t  reserved[10];
    char      platformName[PLATFORM_NAME_LEN]; // unused
    char      rootPath[ROOT_PATH_LEN];         // unused
    char      data[0];
} PrelinkedKernelHeader;

typedef struct platform_info {
    char platformName[PLATFORM_NAME_LEN];
    char rootPath[ROOT_PATH_LEN];
} PlatformInfo;

/*******************************************************************************
*******************************************************************************/

ExitStatus
writeFatFileWithValidation(
                           const char                * filePath,
                           boolean_t                   doValidation,
                           dev_t                       file_dev_t,
                           ino_t                       file_ino_t,
                           CFArrayRef                  fileSlices,
                           CFArrayRef                  fileArchs,
                           mode_t                      fileMode,
                           const struct timeval        fileTimes[2]);

ExitStatus
writeFatFile(
    const char                * filePath,
    CFArrayRef                  fileSlices,
    CFArrayRef                  fileArchs,
    mode_t                      fileMode,
    const struct timeval        fileTimes[2]);
void * mapAndSwapFatHeaderPage(
    int fileDescriptor);
void unmapFatHeaderPage(
    void *headerPage);
struct fat_arch * getFirstFatArch(
    u_char *headerPage);
struct fat_arch * getNextFatArch(
    u_char *headerPage,
    struct fat_arch *lastArch);
struct fat_arch * getFatArchForArchInfo(
    u_char *headerPage,
    const NXArchInfo *archInfo);
const NXArchInfo *
getThinHeaderPageArch(
    const void *headerPage);
ExitStatus
readFatFileArchsWith_fd(
    int                the_fd,
    CFMutableArrayRef * archsOut);
ExitStatus readFatFileArchsWithPath(
    const char        * filePath,
    CFMutableArrayRef * archsOut);
ExitStatus readFatFileArchsWithHeader(
    u_char            * headerPage,
    CFMutableArrayRef * archsOut);
ExitStatus readMachOSlices(
    const char        * filePath,
    CFMutableArrayRef * slicesOut,
    CFMutableArrayRef * archsOut,
    mode_t            * modeOut,
    struct timeval      machOTimesOut[2]);

ExitStatus readMachOSlicesWith_fd(
    int                 the_fd,
    CFMutableArrayRef * slicesOut,
    CFMutableArrayRef * archsOut,
    mode_t            * modeOut,
    struct timeval      machOTimesOut[2]);

CF_RETURNS_RETAINED
CFDataRef  readMachOSliceForArch(
    const char        * filePath,
    const NXArchInfo  * archInfo,
    Boolean             checkArch);

CF_RETURNS_RETAINED
CFDataRef readMachOSliceForArchWith_fd(
    int                   the_fd,
    const NXArchInfo *    archInfo,
    Boolean               checkArch);

CF_RETURNS_RETAINED
CFDataRef readMachOSlice(
    int         fileDescriptor,
    off_t       fileOffset,
    size_t      fileSliceSize);
int readFileAtOffset(
    int             fileDescriptor,
    off_t           fileOffset,
    size_t          fileSize,
    u_char        * buf);
int verifyMachOIsArch(
    const UInt8      * fileBuf,
    size_t              size,
    const NXArchInfo * archInfo);
Boolean supportsFastLibCompression(
    void);
CF_RETURNS_RETAINED
CFDataRef uncompressPrelinkedSlice(
    CFDataRef prelinkImage);
CF_RETURNS_RETAINED
CFDataRef compressPrelinkedSlice(
    uint32_t            compressionType,
    CFDataRef           prelinkImage,
    Boolean             hasRelocs);
ExitStatus writePrelinkedSymbols(
    CFURLRef    symbolDirURL,
    CFArrayRef  prelinkSymbols,
    CFArrayRef  prelinkArchs);
ExitStatus makeDirectoryWithURL(
    CFURLRef dirURL);

#endif /* _KERNELCACHE_H_ */
