/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
 *  modeFirmware.c
 *  bless
 *
 *  Created by Shantonu Sen on 2/22/05.
 *  Copyright 2005-2007 Apple Inc. All Rights Reserved.
 *
 */

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>

#include <sys/mount.h>
#include <sys/stat.h>
#include <fts.h>
#include <dirent.h>

#include <AvailabilityMacros.h>

#include <IOKit/IOKitLib.h>
#include <IOKit/IOBSD.h>
#include <IOKit/storage/IOMedia.h>
#include <IOKit/IOCFSerialize.h>
#include <IOKit/IOCFUnserialize.h>

#include "enums.h"
#include "structs.h"

#include "bless.h"
#include "bless_private.h"
#include "protos.h"

#if USE_DISKARBITRATION
#include <DiskArbitration/DiskArbitration.h>
#endif


int modeFirmware(BLContextPtr context, struct clarg actargs[klast])
{
	int ret = 0;
	
	return ret;
}


