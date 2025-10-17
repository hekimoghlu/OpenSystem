/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
 *  handleDevice.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Thu Dec 6 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: handleDevice.c,v 1.54 2006/07/19 00:15:36 ssen Exp $
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <paths.h>
#include <sys/stat.h>
#include <sys/mount.h>
#include <sys/paths.h>
#include <string.h>

#include <APFS/APFS.h>

#include "enums.h"
#include "structs.h"

#include "bless.h"
#include "bless_private.h"
#include "protos.h"

int modeDevice(BLContextPtr context, struct clarg actargs[klast]) {
    int ret = 0;
	CFDataRef labeldata = NULL;
	CFDataRef labeldata2 = NULL;
    io_object_t devMediaObj;
	
    BLPreBootEnvType	preboot = getPrebootType();
	    
    if(!(geteuid() == 0)) {
		blesscontextprintf(context, kBLLogLevelError,  "Not run as root\n" );
		return 1;
    }


    /* try to grovel the HFS+ catalog and update a label if present */
	if(actargs[klabelfile].present) {
		ret = BLLoadFile(context, actargs[klabelfile].argument, 0, &labeldata);
		if(ret) {
			blesscontextprintf(context, kBLLogLevelError, "Can't load label '%s'\n",
							   actargs[klabelfile].argument);
			return 2;
		}
	} else if(actargs[klabel].present) {
		ret = BLGenerateLabelData(context, actargs[klabel].argument, kBitmapScale_1x, &labeldata);
		if(ret) {
			blesscontextprintf(context, kBLLogLevelError, "Can't render scale 1 label '%s'\n",
							   actargs[klabel].argument);
			return 3;
		}
		ret = BLGenerateLabelData(context, actargs[klabel].argument, kBitmapScale_2x, &labeldata2);
		if(ret) {
			blesscontextprintf(context, kBLLogLevelError, "Can't render scale 2 label '%s'\n",
							   actargs[klabel].argument);
			return 3;
		}
	}
    
    devMediaObj = IOServiceGetMatchingService(kIOMasterPortDefault, IOBSDNameMatching(kIOMasterPortDefault, 0,
                                                                                      actargs[kdevice].argument + strlen(_PATH_DEV)));
    if (!devMediaObj) {
        blesscontextprintf(context, kBLLogLevelError, "Couldn't find I/O Registry information for device %s\n", actargs[kdevice].argument);
        return 4;
    }
    if (IOObjectConformsTo(devMediaObj, APFS_VOLUME_OBJECT)) {
        // This is an APFS volume.  We need to mess with the preboot volume.
        ret = BlessPrebootVolume(context, actargs[kdevice].argument + strlen(_PATH_DEV), NULL, NULL, NULL,
                                 true, actargs);
    }
    IOObjectRelease(devMediaObj);
    if (ret) {
        blesscontextprintf(context, kBLLogLevelError, "Couldn't set bless data in preboot volume for device %s\n", actargs[kdevice].argument);
        return 5;
    }
    		
    /* Set Open Firmware to boot off the specified volume*/
    if(actargs[ksetboot].present) {
        if(preboot == kBLPreBootEnvType_EFI) {

            ret = setefidevice(context, actargs[kdevice].argument + strlen("/dev/"),
                                 actargs[knextonly].present,
                                 actargs[klegacy].present,
                                 actargs[klegacydrivehint].present ? actargs[klegacydrivehint].argument : NULL,
                                 actargs[koptions].present ? actargs[koptions].argument : NULL,
                                 actargs[kshortform].present ? true : false);
        } else {        
            ret = setboot(context, actargs[kdevice].argument, NULL, labeldata);
        }
        
		if(ret) {
			return 3;
		}
	} else if (labeldata) {
		ret = BLSetDiskLabelForDevice(context, actargs[kdevice].argument, labeldata, 1);
		if(ret) {
			blesscontextprintf(context, kBLLogLevelError,  "Error while setting scale 1 label for %s\n", actargs[kdevice].argument );
			return 3;
		}
		if (labeldata2) {
			ret = BLSetDiskLabelForDevice(context, actargs[kdevice].argument, labeldata2, 2);
			if(ret) {
				blesscontextprintf(context, kBLLogLevelError,  "Error while setting scale 2 label for %s\n", actargs[kdevice].argument );
				return 3;
			}
		}
	}

	if (labeldata) CFRelease(labeldata);
	if (labeldata2) CFRelease(labeldata2);

    return 0;
}
