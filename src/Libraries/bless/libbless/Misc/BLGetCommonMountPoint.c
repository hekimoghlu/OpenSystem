/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
 *  BLGetCommonMountPoint.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Tue Apr 17 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLGetCommonMountPoint.c,v 1.14 2006/02/20 22:49:56 ssen Exp $
 *
 */

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/param.h>
#include <sys/mount.h>

#include "bless.h"
#include "bless_private.h"

int BLGetCommonMountPoint(BLContextPtr context, const char * f1,
    const char * f2, char * mountp) {

    struct statfs fsinfo;
    int err;
    char f2mount[MNAMELEN];

    if(f1[0] != '\0') {
		err = statfs(f1, &fsinfo);
      if(err) {
        contextprintf(context, kBLLogLevelError,  "No mount point for %s\n", f1 );
        return 1;
      } else {
	strncpy(mountp, fsinfo.f_mntonname, MNAMELEN-1);
	mountp[MNAMELEN-1] = '\0';
	contextprintf(context, kBLLogLevelVerbose,  "Mount point for %s is %s\n", f1, mountp );
      }
    }

    if(f2[0] != '\0') {
		err = statfs(f2, &fsinfo);
      if(err) {
        contextprintf(context, kBLLogLevelError,  "No mount point for %s\n", f2 );
        return 2;
      } else {
	strncpy(f2mount, fsinfo.f_mntonname, MNAMELEN-1);
	f2mount[MNAMELEN-1] = '\0';
	contextprintf(context, kBLLogLevelVerbose,  "Mount point for %s is %s\n", f2, f2mount );
      }
    }

    /* Now we have the mount points of any folders that were passed
     * in. We must determine:
     * 1) if f1 && f2, find a common mount point or err
     * 2) if f2 && !f1, copy f2mount -> mountp
     * 3) if f1 && !f2, just return success
     */

    if(f2[0] != '\0') {
      /* Case 1, 2 */
      if(f1[0] != '\0') {
	/* Case 1 */
	if(strcmp(mountp, f2mount)) {
	  /* no common */
	  mountp[0] = '\0';
	  return 3;
	} else {
	  /* yay common */
	  return 0;
	}
      } else {
	/* Case 2 */

	/* We know each buffer is <MNAMELEN and 0-terminated */
	strncpy(mountp, f2mount, MNAMELEN);
	return 0;
      }
    } else {
      /* Case 3 */
      return 0;
    }

    contextprintf(context, kBLLogLevelError,  "No folders specified" );
    return 4;
}
