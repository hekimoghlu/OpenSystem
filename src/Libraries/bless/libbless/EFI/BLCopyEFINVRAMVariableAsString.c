/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
 *  BLCopyEFINVRAMVariableAsString.c
 *  bless
 *
 *  Created by Shantonu Sen on 12/2/05.
 *  Copyright 2005-2007 Apple Inc. All Rights Reserved.
 *
 */

#include <IOKit/IOKitLib.h>
#include <IOKit/IOKitKeys.h>

#include "bless.h"
#include "bless_private.h"

int BLCopyEFINVRAMVariableAsString(BLContextPtr context,
                                   CFStringRef  name,
                                   CFStringRef *value)
{
    
    io_registry_entry_t optionsNode = 0;
    char            cStr[1024];
    CFTypeRef       valRef;
    CFStringRef     stringRef;
    
    *value = NULL;
    
    optionsNode = IORegistryEntryFromPath(kIOMasterPortDefault, kIODeviceTreePlane ":/options");
    
    if(IO_OBJECT_NULL == optionsNode) {
        contextprintf(context, kBLLogLevelError,  "Could not find " kIODeviceTreePlane ":/options\n");
        return 1;
    }
    
    valRef = IORegistryEntryCreateCFProperty(optionsNode, name, kCFAllocatorDefault, 0);
    IOObjectRelease(optionsNode);
    
    if(valRef == NULL)
        return 0;
    
    if(CFGetTypeID(valRef) == CFStringGetTypeID()) {
        if(!CFStringGetCString(valRef, cStr, sizeof(cStr), kCFStringEncodingUTF8)) {
            contextprintf(context, kBLLogLevelVerbose,
                               "Could not interpret NVRAM variable as UTF-8 string. Ignoring...\n");
            cStr[0] = '\0';
        }
    } else if(CFGetTypeID(valRef) == CFDataGetTypeID()) {
        const UInt8 *ptr = CFDataGetBytePtr(valRef);
        CFIndex len = CFDataGetLength(valRef);
        
        if(len > sizeof(cStr)-1)
            len = sizeof(cStr)-1;
        
        memcpy(cStr, (char *)ptr, len);
        cStr[len] = '\0';
        
    } else {
        contextprintf(context, kBLLogLevelError,  "Could not interpret NVRAM variable. Ignoring...\n");
        cStr[0] = '\0';
    }
    
    CFRelease(valRef);
    
    stringRef = CFStringCreateWithCString(kCFAllocatorDefault, cStr, kCFStringEncodingUTF8);
    if(stringRef == NULL) {
        return 2;
    }
    
    *value = stringRef;
    
    return 0;
}
