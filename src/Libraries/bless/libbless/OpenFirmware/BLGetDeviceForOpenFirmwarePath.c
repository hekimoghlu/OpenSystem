/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
 *  BLGetDeviceForOpenFirmwarePath.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Thu Jan 24 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLGetDeviceForOpenFirmwarePath.c,v 1.22 2006/02/20 22:49:57 ssen Exp $
 *
 */
 
#import <mach/mach_error.h>
#import <IOKit/IOKitLib.h>
#import <IOKit/IOBSD.h>
#import <IOKit/IOKitKeys.h>
#import <IOKit/storage/IOMedia.h>

#import <CoreFoundation/CoreFoundation.h>

#include <string.h>
#include <stdlib.h>
#include <sys/param.h>
#include <sys/mount.h>

#include <sys/stat.h>
 
#include "bless.h"
#include "bless_private.h"


int BLGetDeviceForOpenFirmwarePath(BLContextPtr context, const char * ofstring, char * mntfrm) {
	// OpenFirmware not supported anymore
    return 1;
}

