/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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

//
//  ApplePIODMADebug.cpp
//  ApplePIODMA
//
//  Created by Kevin Strasberg on 6/29/20.
//

#include <IOKit/apiodma/ApplePIODMADebug.h>

uint32_t applePIODMAgetDebugLoggingMask(const char* bootArg)
{
    uint32_t localDebugMask = 0;
    PE_parse_boot_argn(bootArg, &localDebugMask, sizeof(localDebugMask));

    // Include special masks that apply to all boot-args
    localDebugMask |= kApplePIODMADebugLoggingAlways;

    return localDebugMask;
}

uint32_t applePIODMAgetDebugLoggingMaskForMetaClass(const OSMetaClass* metaClass, const OSMetaClass* stopClass, const char* location)
{
    uint32_t result       = 0;
    char     bootArg[256] = { 0 };

    while(   metaClass != NULL
          && metaClass != stopClass)
    {
        if(location != NULL)
        {
            snprintf(bootArg, 256, "%s@%s-debug", metaClass->getClassName(), location);
        }
        else
        {
            snprintf(bootArg, 256, "%s-debug", metaClass->getClassName());
        }

        result |= applePIODMAgetDebugLoggingMask(bootArg);

        metaClass = metaClass->getSuperClass();
    }

    return result;
}
