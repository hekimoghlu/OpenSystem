/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
 *  BLIsOpenFirmwarePresent.c
 *  bless
 *
 *  Created by Shantonu Sen on Tue Jul 22 2003.
 *  Copyright (c) 2003-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLIsOpenFirmwarePresent.c,v 1.11 2006/02/20 22:49:57 ssen Exp $
 *
 */

#include <stdlib.h>
#include <unistd.h>
#include <sys/param.h>

#include <mach/mach_error.h>

#include <IOKit/IOKitLib.h>
#include <IOKit/IOKitKeys.h>

#include <CoreFoundation/CoreFoundation.h>

#include "bless.h"
#include "bless_private.h"

#define kBootRomPath "/openprom"

int BLIsOpenFirmwarePresent(BLContextPtr context) {

  BLPreBootEnvType preboot;
  int ret;

  ret = BLGetPreBootEnvironmentType(context, &preboot);

  // on error, assume no OF
  if(ret) return 0;

  if(preboot == kBLPreBootEnvType_OpenFirmware)
    return 1;
  else
    return 0;

}
