/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
 *  BLLookupFileIDOnMount.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Thu Apr 19 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <hfs/hfs_format.h>
#include <sys/param.h>
#include <sys/mount.h>
#include <sys/attr.h>
#include <System/sys/fsgetpath.h>

#include "bless.h"
#include "bless_private.h"

struct cataloginfo {
  attrreference_t name;
  fsid_t volid;
  fsobj_id_t objectid;
  fsobj_id_t parentid;
  char namebuffer[NAME_MAX];
};

struct cataloginforeturn {
  uint32_t length;
  struct cataloginfo c;
};


static int lookupIDOnVolID(uint32_t volid, uint32_t fileID, char *out);

int BLLookupFileIDOnMount(BLContextPtr context, const char *mount, uint32_t fileID, char *out) {
    struct attrlist alist;
    struct cataloginforeturn catinfo;
    int err;

    uint32_t volid;
    char relpath[MAXPATHLEN];
	
	out[0] = '\0';

    if (fileID < kHFSRootFolderID) {
		return ENOENT;
    }

    alist.bitmapcount = 5;
    alist.commonattr = ATTR_CMN_NAME | ATTR_CMN_FSID | ATTR_CMN_OBJID | ATTR_CMN_PAROBJID;
    alist.volattr = 0;
    alist.dirattr = 0;
    alist.fileattr = 0;
    alist.forkattr = 0;
    
	err = getattrlist(mount, &alist, &catinfo, sizeof(catinfo), 0);
    if (err) {
        return errno;
    }

    volid = (uint32_t)catinfo.c.volid.val[0];

    err = lookupIDOnVolID(volid, fileID, relpath);
    if (err) {
        return err;
    }

    if(strcmp(mount, "/")) {
        /* If the mount point is not '/', prefix by mount */
        snprintf(out, MAXPATHLEN, "%s/%s", mount, relpath);
    } else {
        snprintf(out, MAXPATHLEN, "/%s", relpath);
    }

    return 0;
}




int BLLookupFileIDOnMount64(BLContextPtr context, const char *mountpoint, uint64_t fileID, char *out, int bufsize)
{
    int err;
    
    struct statfs   sfs;
    
    if (statfs(mountpoint, &sfs) < 0) {
        return errno;
    }
    
	if (__builtin_available(macOS 10.13, *)) {
		err = (int)fsgetpath(out, bufsize, &sfs.f_fsid, fileID);
		if (err < 0) return errno;
	} else {
		err = ENOTSUP;
	}
	
    return 0;
}



static int lookupIDOnVolID(uint32_t volid, uint32_t fileID, char *out) {

    char *bp;

    uint32_t dirID = fileID; /* to initialize loop */
    char volpath[MAXPATHLEN];

    struct attrlist alist;
    struct cataloginforeturn catinfo;
    int err;

    out[0] = '\0';

    if (fileID <= 2) {
        return 0;
    }

    /* Now for the recursive step
     * 1. getattr on /.vol/volid/dirID
     * 2. get the name.
     * 3. set dirID = parentID
     * 4. go to 1)
     * 5. exit when dirID == 2
    */


    /* bp will hold our current position. Work from the end
     * of the buffer until the beginning */
    bp = &(out[MAXPATHLEN-1]);
    *bp = '\0';

    while(dirID != kHFSRootFolderID) {
        char *nameptr;
        size_t namelen;
        snprintf(volpath, sizeof(volpath), "/.vol/%u/%u", volid, dirID);
        alist.bitmapcount = 5;
        alist.commonattr = ATTR_CMN_NAME | ATTR_CMN_FSID | ATTR_CMN_OBJID | ATTR_CMN_PAROBJID;
        alist.volattr = 0;
        alist.dirattr = 0;
        alist.fileattr = 0;
        alist.forkattr = 0;
        
        err = getattrlist(volpath, &alist, &catinfo, sizeof(catinfo), 0);
        if (err) {
            return errno;
        }

        dirID = (uint32_t)catinfo.c.parentid.fid_objno;
        nameptr = (char *)(&catinfo.c.name) + catinfo.c.name.attr_dataoffset;
        namelen = strlen(nameptr); /* move bp by this many and copy */
		
		/* make sure we don't overwrite beginning of buffer */
		if (bp - out < namelen) {
			return ENAMETOOLONG;
		}
		
        bp -= namelen;
        strncpy(bp, nameptr, namelen); /* ignore trailing \0 */
		
		if (dirID != kHFSRootFolderID /* 2 */) {
			/* make sure we don't overwrite beginning of buffer */
			if (!(bp > out)) {
				return ENAMETOOLONG;
			}
			bp--;
			*bp = '/';
		}
	} // while dirID != 2

    memmove(out, bp, strlen(bp)+1);
    return 0;
}
