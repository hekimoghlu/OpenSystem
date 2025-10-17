/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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

#include <fcntl.h>
#include <util.h>
#include <unistd.h>
#include <string.h>
#include <sys/mount.h>
#include <uuid/uuid.h>
#include <IOKit/IOBSD.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/storage/IOMedia.h>

#include "check.h"

/*
 * Given a uuid string, look up the BSD device and open it.
 * This code comes from DanM.
 *
 * Essentially, it is given a UUID string (from the journal header),
 * and then looks it up via IOKit.  From there, it then gets the
 * BSD name (e.g., /dev/dsik3), and opens it read-only.
 *
 * It returns the file descriptor, or -1 on error.
 */
int
OpenDeviceByUUID(void *uuidp, char **namep)
{
    char devname[ MAXPATHLEN ];
    CFStringRef devname_string;
    int fd = -1;
    CFMutableDictionaryRef matching;
    io_service_t media;
    uuid_string_t uuid_cstring;
    CFStringRef uuid_string;

    memcpy(&uuid_cstring, uuidp, sizeof(uuid_cstring));

    uuid_string = CFStringCreateWithCString( kCFAllocatorDefault, uuid_cstring, kCFStringEncodingUTF8 );
    if ( uuid_string ) {
        matching = IOServiceMatching( kIOMediaClass );
        if ( matching ) {
            CFDictionarySetValue( matching, CFSTR( kIOMediaUUIDKey ), uuid_string );
            media = IOServiceGetMatchingService( kIOMainPortDefault, matching );
            if ( media ) {
                devname_string = IORegistryEntryCreateCFProperty( media, CFSTR( kIOBSDNameKey ), kCFAllocatorDefault, 0 );
                if ( devname_string ) {
                    if ( CFStringGetCString( devname_string, devname, sizeof( devname ), kCFStringEncodingUTF8 ) ) {
			if (state.debug)
				fsck_print(ctx, LOG_TYPE_INFO, "external journal device name = `%s'\n", devname);

                        fd = opendev( devname, O_RDONLY, 0, NULL );
			if (fd != -1 && namep != NULL) {
				*namep = strdup(devname);
			}
                    }
                    CFRelease( devname_string );
                }
                IOObjectRelease( media );
            }
            /* do not CFRelease( matching ); */
        }
        CFRelease( uuid_string );
    }

    return fd;
}
