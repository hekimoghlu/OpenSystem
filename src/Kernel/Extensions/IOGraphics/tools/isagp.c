/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#include <CoreFoundation/CoreFoundation.h>
#include <ApplicationServices/ApplicationServices.h>
#include <IOKit/graphics/IOGraphicsLib.h>
#include <stdlib.h>
#include <stdio.h>


int main(int argc, char * argv[])
{
    io_service_t        device;
    io_service_t        framebuffer;
    io_service_t        accelerator;
    UInt32              framebufferIndex;
    CGError             err;
    int                 i;
    CGDisplayCount      max;
    CGDirectDisplayID   displayIDs[8];

    err = CGGetOnlineDisplayList(8, displayIDs, &max);
    if(err != kCGErrorSuccess)
        exit(1);
    if(max > 8)
        max = 8;

    for(i = 0; i < max; i++ ) {

        framebuffer = CGDisplayIOServicePort(displayIDs[i]);

        err = IOAccelFindAccelerator(framebuffer, &accelerator, &framebufferIndex);
        if(kIOReturnSuccess != err)
            continue;

        err = IORegistryEntryGetParentEntry(accelerator, kIOServicePlane, &device);
        IOObjectRelease(accelerator);
        if(kIOReturnSuccess != err)
            continue;

        printf("Display ID %p ", displayIDs[i]);
        if(IOObjectConformsTo(device, "IOAGPDevice"))
            printf("is");
        else
            printf("isn't");
        printf(" agp\n");

        IOObjectRelease(device);
    }
    
    exit(0);
    return(0);
}

