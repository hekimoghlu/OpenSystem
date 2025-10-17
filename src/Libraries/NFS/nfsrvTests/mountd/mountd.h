/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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
/*
 * Copyright (c) 1992, 1993, 1994
 *    The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Rick Macklem at The University of Guelph.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the University of
 *    California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */


#ifndef NFSRV_TESTS_MOUNTD
#define NFSRV_TESTS_MOUNTD

#include <sys/mount.h>

void doMountSetUp(void);
void doMountSetUpWithArgs(const char **nfsdArgs, int nfsdArgsSize, const char **confValues, int confValuesSize);
void doMountTearDown(void);

extern char exportsPath[PATH_MAX];
extern char confPath[PATH_MAX];
extern CLIENT *mclnt;

const char *getRootDir(void);
const char *getDestPath(void);
const char *getDestReadOnlyPath(void);
const char *getLocalMountedPath(void);
const char *getLocalMountedReadOnlyPath(void);
fhandle_t *doMountAndVerify(const char *dir, char *sec_mech);

#define CREATE_CLIENT_FAILURE   0x1
#define CREATE_SOCKET           0x2
#define CREATE_NFS_V2           0x4

int createClientForMountProtocol(int socketFamily, int socketType, int authType, int flags);
CLIENT *createClientForNFSProtocol(int socketFamily, int socketType, int authType, int flags, int *sockp);
CLIENT *createClientForProtocol(const char *host, int socketFamily, int socketType, int authType, unsigned int program, unsigned int version, int flags, int *sockp);

#define REMOVE_FILE     0
#define REMOVE_DIR      AT_REMOVEDIR

#define LOCALHOST4 "127.0.0.1"
#define LOCALHOST6 "::1"

#define UNKNOWNUID ((uid_t)99)
#define UNKNOWNGID ((gid_t)99)

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(A) (sizeof(A) / sizeof(A[0]))
#endif

int removeFromPath(char *file, int dirFD, int fileFD, int mode);
int createFileInPath(const char *dir, char *file, int *dirFDp, int *fileFDp);

#endif /* NFSRV_TESTS_MOUNTD */
