/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
 *  BLMiscUtilities.c
 *  bless
 *
 *  Created by Shantonu Sen on Sat Apr 19 2003.
 *  Copyright (c) 2003-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLMiscUtilities.c,v 1.7 2006/02/20 22:49:56 ssen Exp $
 *
 */

#include "bless.h"
#include "bless_private.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mount.h>

char * blostype2string(uint32_t type, char *buf, uint32_t bufSize)
{
    bzero(buf, bufSize);
    if(type == 0 || bufSize < 5)
        return buf;

    snprintf(buf, bufSize, "%c%c%c%c",
	    (type >> 24)&0xFF,
	    (type >> 16)&0xFF,
	    (type >> 8)&0xFF,
	    (type >> 0)&0xFF);

    return buf;    
}

int blsustatfs(const char *path, struct statfs *buf)
{
    int ret;
    struct stat sb;
    int flag;
    char *dev = NULL;
    
    ret = statfs(path, buf);    
    if(ret)
        return ret;
	
	
#ifdef AT_REALDEV
	if (__builtin_available(macOS 10.15, *)) {
		flag = AT_REALDEV;
	} else {
		flag = 0;
	}
#else
    flag = 0;
#endif /* AT_REALDEV */
    ret = fstatat(AT_FDCWD, path, &sb, flag);
    if(ret)
        return ret;
    
    // figure out the true device we live on
    dev = devname(sb.st_dev, S_IFBLK);
    if(dev == NULL) {
        errno = ENOENT;
        return -1;
    }
    
    snprintf(buf->f_mntfromname, sizeof(buf->f_mntfromname), "/dev/%s", dev);
    
    return 0;
}

