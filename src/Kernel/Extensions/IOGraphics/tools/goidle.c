/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include <libgen.h>
#include <stdlib.h>
#include <stdio.h>
#include <sysexits.h>
#include <unistd.h>

#include <mach/mach_port.h>
#include <mach/mach_interface.h>
#include <mach/mach_init.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>


static const char *sCmdName;

static void usage(const char *errMsg)
{
    if (errMsg && *errMsg)
        fprintf(stderr, "%s: %s\n", sCmdName, errMsg);
    fprintf(stderr,
"usage: %s [-i|-d] [0|<seconds>]\n"
"where\n"
"    -i  Idle request (default).\n"
"The argument sent to the kernel is True by default, otherwise the seconds\n"
"argument is interpreted a 0 -> False, and n is the time in seconds to ignore\n"
"user interface events.\n",
        sCmdName);

    exit(EX_USAGE);
}

int
main(int argc, char **argv)
{
    sCmdName = basename(argv[0]);

    int iflag = 1;  // RequestIdle by default
    int dflag = 0;

    int ch;
    while ((ch = getopt(argc, argv, "di")) != -1) {
        switch (ch) {
        case 'i': iflag = 1; dflag = 0; break;
        case '?':
        default:  usage(""); break;
        }
    }
    argc -= optind;
    argv += optind;

    CFTypeRef obj = CFRetain(kCFBooleanTrue);
    if (argc >= 1) {
        int32_t num = 1000 * strtol(argv[0], NULL, 0);
        if (!num)
            obj = CFRetain(kCFBooleanFalse);
        else if (iflag)
            obj = CFNumberCreate(NULL, kCFNumberSInt32Type, &num);
    }

    const char *request = "Internal error";
    if (iflag)
        request = "IORequestIdle";
    CFStringRef requestStr = CFStringCreateWithCString(
                                          NULL, request, kCFStringEncodingUTF8);

    io_registry_entry_t regEntry = IORegistryEntryFromPath(
            kIOMasterPortDefault,
            kIOServicePlane ":/IOResources/IODisplayWrangler");

    int32_t objValue = -1;
    const char *objType = "Unknown";
    if (CFGetTypeID(obj) == CFBooleanGetTypeID()) {
        objType = "bool"; objValue = CFBooleanGetValue(obj);
    } else if (CFGetTypeID(obj) == CFNumberGetTypeID()) {
        objType = "number";
        CFNumberGetValue(obj, kCFNumberSInt32Type, &objValue);
    }

    kern_return_t err = IORegistryEntrySetCFProperty(regEntry, requestStr, obj);
    printf("IORegistryEntrySetCFProperty(%s: %s(%d)) %#x\n",
            request, objType, objValue, err);

    // No need to release we're about to exit
    // CFRelease(requestStr);
    // CFRelease(obj);
    // IOObjectRelease(regEntry);

    return (err) ? EX_OSERR : EX_OK;
}
