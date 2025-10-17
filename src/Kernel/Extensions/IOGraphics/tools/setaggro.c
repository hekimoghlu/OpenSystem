/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
#include <IOKit/graphics/IOGraphicsTypesPrivate.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef sub_iokit_graphics
#define sub_iokit_graphics           err_sub(5)
#endif

#ifndef kIOFBLowPowerAggressiveness
#define kIOFBLowPowerAggressiveness     iokit_family_err(sub_iokit_graphics, 1)
#endif

#ifndef kIODisplayDimAggressiveness
#define kIODisplayDimAggressiveness     iokit_family_err(sub_iokit_graphics, 3)
#endif

int main(int argc, char * argv[])
{
    kern_return_t err;
    io_connect_t  connect;
    unsigned long value;

    if (argc < 2)
    {
        fprintf(stderr, "%s value\n", argv[0]);
        return (1);
    }

    connect = IOPMFindPowerManagement(kIOMasterPortDefault);
    if (!connect) 
    {
        fprintf(stderr, "IOPMFindPowerManagement(%x)\n", err);
        return (1);
    }

    value = strtol(argv[1], 0, 0);

#if 1
    err = IOPMSetAggressiveness( connect, kIOFBLowPowerAggressiveness, value );
    fprintf(stderr, "IOPMSetAggressiveness(kIOFBLowPowerAggressiveness, %lx) result %x\n", value, err);
#else
    err = IOPMSetAggressiveness( connect, kIODisplayDimAggressiveness, value );
    fprintf(stderr, "IOPMSetAggressiveness(kIODisplayDimAggressiveness, %lx) result %x\n", value, err);
#endif    
    IOServiceClose(connect);

    exit (0);
    return (0);
}

