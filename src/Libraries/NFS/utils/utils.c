/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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

#include "utils.h"

#include <err.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/syslog.h>

#include <TargetConditionals.h>

#if TARGET_OS_OSX
#include <IOKit/kext/KextManager.h>
#endif /* TARGET_OS_OSX */

#define NFSCLIENT_KEXT_TMPFILE "/private/var/tmp/nfsclient_kext.tmp"

void
load_nfs_kext(void)
{
#if TARGET_OS_OSX
    int fd;
    struct vfsconf vfc;
    struct stat filestat = {};
    kern_return_t status;

    setlocale(LC_CTYPE, "");
    if (getvfsbyname("nfs", &vfc) != 0) {
        /* Need to load the kext */
        status = KextManagerLoadKextWithIdentifier(CFSTR("com.apple.filesystems.nfs"), NULL);
        if (status != KERN_SUCCESS) {
            errx(EIO, "Loading com.apple.filesystems.nfs status = 0x%x", status);
        }

        /* Trigger launchd to parse nfs.conf */
        if ((fd = open(NFSCLIENT_KEXT_TMPFILE, O_WRONLY | O_CREAT | O_TRUNC, 666)) < 0) {
            warnx("Cannot open/create %s, nfs conf will not be parsed\n", NFSCLIENT_KEXT_TMPFILE);
        } else {
            close(fd);
            for (int i = 0; stat(NFSCLIENT_KEXT_TMPFILE, &filestat) != 0 && i < 5; i++) {
                /* Wait for nfs.conf to be parsed by launchd */
                sleep(1);
            }
        }
    }
#endif /* TARGET_OS_OSX */
}

void
unlink_kext_tmpfile(void)
{
#if TARGET_OS_OSX
    if (unlink(NFSCLIENT_KEXT_TMPFILE) < 0) {
        warnx("Cannot unlink %s\n", NFSCLIENT_KEXT_TMPFILE);
    }
#endif /* TARGET_OS_OSX */
}

#include "utils.h"
