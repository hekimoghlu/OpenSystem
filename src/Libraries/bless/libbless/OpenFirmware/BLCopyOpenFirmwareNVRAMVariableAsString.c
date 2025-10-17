/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
 *  BLCopyOpenFirmwareNVRAMVariableAsString.c
 *  bless
 *
 *  Created by Shantonu Sen on 7/7/07.
 *  Copyright 2007 Apple Inc. All Rights Reserved.
 *
 */

#include <IOKit/IOKitLib.h>
#include <IOKit/IOKitKeys.h>

#include "bless.h"
#include "bless_private.h"

int BLCopyOpenFirmwareNVRAMVariableAsString(BLContextPtr context,
                                           CFStringRef  name,
                                           CFStringRef *value)
{
    // actually, the same thing as for EFI
    return BLCopyEFINVRAMVariableAsString(context, name, value);
}
