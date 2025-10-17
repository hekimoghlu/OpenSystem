/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
 *  unbless.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Sun Mar 6, 2005.
 *  Copyright (c) 2005-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: unbless.c,v 1.2 2005/09/12 22:09:06 ssen Exp $
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mount.h>
#include <sys/param.h>
#include <err.h>

#include "bless.h"

int unbless(char *mountpoint);

int main(int argc, char *argv[]) {

  char *mntpnt = NULL;
  struct statfs sb;
  int ret;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s /Volumes/foo\n", getprogname());
    exit(1);
  }

  mntpnt = argv[1];

  ret = statfs(mntpnt, &sb);
  if(ret)
    err(1, "statfs(%s)", mntpnt);

  if(0 != strcmp(mntpnt, sb.f_mntonname))
    errx(1, "Path is not a mount point");

  ret = unbless(mntpnt);

  return ret;
}


int unbless(char *mountpoint) {
	
    int ret;
    int isHFS;
    uint32_t oldwords[8];
		
    ret = BLIsMountHFS(NULL, mountpoint, &isHFS);
    if(ret)
      errx(1, "Could not determine filesystem of %s", mountpoint);

    if(!isHFS)
      errx(1, "%s is not HFS+", mountpoint);
    
    ret = BLGetVolumeFinderInfo(NULL, mountpoint, oldwords);
    if(ret)
      err(1, "Could not get finder info for %s", mountpoint);
		
    oldwords[0] = 0;
    oldwords[1] = 0;
    oldwords[2] = 0;
    oldwords[3] = 0;
    oldwords[5] = 0;
		
    /* bless! bless */
    
    ret = BLSetVolumeFinderInfo(NULL,  mountpoint, oldwords);
    if(ret)
      err(1, "Can't set finder info for %s", mountpoint);
	
    return 0;
}

