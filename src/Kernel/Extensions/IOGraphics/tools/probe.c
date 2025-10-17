/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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
#include <IOKit/graphics/IOGraphicsTypesPrivate.h>
#include <stdlib.h>
#include <stdio.h>


int main(int argc, char * argv[])
{
    io_service_t        service;
    CGError             err;
    int                 i;
    CGDisplayCount      max;
    CGDirectDisplayID   displayIDs[8];
    uint32_t            mask;
    IOOptionBits        options;
    CFNumberRef         num;
    SInt32              value;

    // usage: probe [options=kIOFBUserRequestProbe] [mask=0xffffffff]
    // usage: rotate <degrees> [mask=0xffffffff]

    err = CGGetOnlineDisplayList(8, displayIDs, &max);
    if(err != kCGErrorSuccess)
        exit(1);
    if(max > 8)
        max = 8;

    if( argc < 2)
        options = kIOFBUserRequestProbe;
    else
        options = strtol( argv[1], 0, 0 );

    if (strstr(argv[0], "rotate"))
    {
        switch (options)
        {
            case 90:
              options = kIOFBSetTransform | (kIOScaleRotate90 << 16);
              break;
            case 180:
              options = kIOFBSetTransform | (kIOScaleRotate180 << 16);
              break;
            case 270:
              options = kIOFBSetTransform | (kIOScaleRotate270 << 16);
              break;
            case 0:
            default:
              options = kIOFBSetTransform | (kIOScaleRotate0 << 16);
              break;
        }
    }

    if( argc < 3)
        mask = 0xffffffff;
    else
        mask = strtol( argv[2], 0, 0 );

    for(i = 0; i < max; i++ )
    {
        if (!(mask & (1 << i)))
            continue;

        service = CGDisplayIOServicePort(displayIDs[i]);


        num = (CFNumberRef) IORegistryEntryCreateCFProperty( service, 
                                                                CFSTR(kIOFBTransformKey),
                                                                kCFAllocatorDefault, kNilOptions);
        if (num)
          CFNumberGetValue( num, kCFNumberSInt32Type, (SInt32 *) &value );
        else
          value = 0;

        value &= kIOScaleRotateFlags;

        printf("Display %#x: current transform: ", displayIDs[i]);

        switch (value)
        {
            case kIOScaleRotate90:
              printf("90\n");
              break;

            case kIOScaleRotate180:
              printf("180\n");
              break;

            case kIOScaleRotate270:
              printf("270\n");
              break;

            case kIOScaleRotate0:
            default:
              printf("0\n");
              break;
        }


        num = (CFNumberRef) IORegistryEntryCreateCFProperty( service, 
                                                                CFSTR(kIOFBProbeOptionsKey),
                                                                kCFAllocatorDefault, kNilOptions);
        if (num)
          CFNumberGetValue( num, kCFNumberSInt32Type, (SInt32 *) &value );
        else
          value = 0;
        printf("Display %#x: does %ssupport kIOFBSetTransform\n", displayIDs[i], value & kIOFBSetTransform ? "" : "not ");

        if (value & kIOFBSetTransform)
        {
          err = IOServiceRequestProbe(service, options );
          printf("Display %#x: IOServiceRequestProbe(%d)\n", displayIDs[i], err);
        }
    }
    
    exit(0);
    return(0);
}

