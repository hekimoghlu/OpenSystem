/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
 *  BLSetOpenFirmwareBootDeviceForMountPoint.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Tue Apr 17 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLSetOpenFirmwareBootDeviceForMountPoint.c,v 1.14 2006/02/20 22:49:57 ssen Exp $
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/stat.h>

#include "bless.h"
#include "bless_private.h"

#define NVRAM "/usr/sbin/nvram"

int BLSetOpenFirmwareBootDeviceForMountPoint(BLContextPtr context, const char * mountpoint) {
    char           mntfrm[MAXPATHLEN];
    int err;
    struct stat sb;

    err = stat(mountpoint, &sb);
    if(err) {
      contextprintf(context, kBLLogLevelError,  "Can't stat mount point %s\n", mountpoint );
      return 1;
    }

    if(devname(sb.st_dev, S_IFBLK) == NULL) {
            return 2;
    }

    snprintf(mntfrm, MAXPATHLEN, "/dev/%s", devname(sb.st_dev, S_IFBLK));
    return BLSetOpenFirmwareBootDevice(context, mntfrm);
}
    
