/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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
#include "optical.h"

#include <paths.h>
#include <string.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/storage/IOMedia.h>
#include <IOKit/storage/IOBDMedia.h>
#include <IOKit/storage/IOCDMedia.h>
#include <IOKit/storage/IODVDMedia.h>

static io_service_t
__io_media_copy_whole_media(io_service_t media)
{
    io_service_t parent;
    CFTypeRef property;

    IOObjectRetain(media);

    while (media) {
        if (IOObjectConformsTo(media, kIOMediaClass)) {
            property = IORegistryEntryCreateCFProperty(media, CFSTR(kIOMediaWholeKey), kCFAllocatorDefault, 0);
            if (property) {
                CFRelease(property);
                if (property == kCFBooleanTrue)
                    break;
            }
        }
        parent = IO_OBJECT_NULL;
        IORegistryEntryGetParentEntry(media, kIOServicePlane, &parent);
        IOObjectRelease(media);
        media = parent;
    }

    return media;
}

static io_service_t
__io_media_create_from_bsd_name(const char *name)
{
    if (!strncmp(_PATH_DEV, name, strlen(_PATH_DEV)))
        name += strlen(_PATH_DEV);

    return IOServiceGetMatchingService(kIOMainPortDefault, IOBSDNameMatching(kIOMainPortDefault, 0, name));
}

static int
__io_media_is_writable(io_service_t media)
{
    int writable = 0;
    CFTypeRef property;

    property = IORegistryEntryCreateCFProperty(media, CFSTR(kIOMediaWritableKey), kCFAllocatorDefault, 0);
    if (property) {
        CFRelease(property);
        if (property == kCFBooleanTrue) {
            writable = _OPTICAL_WRITABLE_SECTOR;

            if (IOObjectConformsTo(media, kIOBDMediaClass)) {
                property = IORegistryEntryCreateCFProperty(media, CFSTR(kIOBDMediaTypeKey), kCFAllocatorDefault, 0);
                if (property) {
                    if (CFEqual(property, CFSTR(kIOBDMediaTypeR)))
                        writable = _OPTICAL_WRITABLE_PACKET | _OPTICAL_WRITABLE_ONCE;   /* BD-R */
                    else if (CFEqual(property, CFSTR(kIOBDMediaTypeRE)))
                        writable = _OPTICAL_WRITABLE_PACKET;                           /* BD-RE */
                    CFRelease(property);
                }
            } else if (IOObjectConformsTo(media, kIOCDMediaClass)) {
                property = IORegistryEntryCreateCFProperty(media, CFSTR(kIOCDMediaTypeKey), kCFAllocatorDefault, 0);
                if (property) {
                    if (CFEqual(property, CFSTR(kIOCDMediaTypeR)))
                        writable = _OPTICAL_WRITABLE_PACKET | _OPTICAL_WRITABLE_ONCE;   /* CD-R */
                    else if (CFEqual(property, CFSTR(kIOCDMediaTypeRW)))
                        writable = _OPTICAL_WRITABLE_PACKET;                            /* CD-RW */

                    CFRelease(property);
                }
            } else if (IOObjectConformsTo(media, kIODVDMediaClass)) {
                property = IORegistryEntryCreateCFProperty(media, CFSTR(kIODVDMediaTypeKey), kCFAllocatorDefault, 0);
                if (property) {
                    if (CFEqual(property, CFSTR(kIODVDMediaTypeR)))
                        writable = _OPTICAL_WRITABLE_PACKET | _OPTICAL_WRITABLE_ONCE;   /* DVD-R */
                    else if (CFEqual(property, CFSTR(kIODVDMediaTypeRW)))
                        writable = _OPTICAL_WRITABLE_PACKET;                            /* DVD-RW */
                    else if (CFEqual(property, CFSTR(kIODVDMediaTypePlusR)))
                        writable = _OPTICAL_WRITABLE_PACKET | _OPTICAL_WRITABLE_ONCE;   /* DVD+R */
                    else if (CFEqual(property, CFSTR(kIODVDMediaTypePlusRW)))
                        writable = _OPTICAL_WRITABLE_PACKET;                            /* DVD+RW */
                    else if (CFEqual(property, CFSTR(kIODVDMediaTypeHDR)))
                        writable = _OPTICAL_WRITABLE_PACKET | _OPTICAL_WRITABLE_ONCE;   /* HD DVD-R */

                    CFRelease(property);
                }
            }
        }
    }

    return writable;
}

int
_optical_is_writable(const char *dev)
{
    int writable = 0;
    io_service_t media;
    io_service_t whole;

    media = __io_media_create_from_bsd_name(dev);
    if (media) {
        writable = __io_media_is_writable(media);
        if (writable) {
            whole = __io_media_copy_whole_media(media);
            if (whole) {
                writable = __io_media_is_writable(whole);

                IOObjectRelease(whole);
            }
        }
        IOObjectRelease(media);
    }

    return writable;
}
