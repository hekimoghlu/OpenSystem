/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#include <assert.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOKitKeys.h>

int main(int argc, char **argv)
{
    io_registry_entry_t    service;
    io_connect_t           connect;
    SInt32                 id1, id2, id3;
    kern_return_t          status;

    service = IORegistryEntryFromPath(kIOMasterPortDefault, 
                                    kIOServicePlane ":/IOResources/IODisplayWrangler");
    assert(service);
    if (service) 
    {
        status = IOServiceOpen(service, mach_task_self(), 0, &connect);
        IOObjectRelease(service);
        assert(kIOReturnSuccess == status);

    }

    enum { kAlloc, kFree };
enum {
    kIOAccelSpecificID          = 0x00000001
};


    status = IOConnectMethodScalarIScalarO(connect, kAlloc, 2, 1, kNilOptions, 0, &id1);
    assert(kIOReturnSuccess == status);
    printf("ID: %x\n", id1);
    status = IOConnectMethodScalarIScalarO(connect, kFree, 2, 0, kNilOptions, id1);
    assert(kIOReturnSuccess == status);
    status = IOConnectMethodScalarIScalarO(connect, kAlloc, 2, 1, kNilOptions, 0, &id1);
    assert(kIOReturnSuccess == status);
    printf("ID: %x\n", id1);


    status = IOConnectMethodScalarIScalarO(connect, kAlloc, 2, 1, kIOAccelSpecificID, 53, &id2);
    assert(kIOReturnSuccess == status);
    printf("ID: %x\n", id2);


    status = IOConnectMethodScalarIScalarO(connect, kFree, 2, 0, kNilOptions, id1);
    assert(kIOReturnSuccess == status);
    printf("free ID: %d\n", id1);

    status = IOConnectMethodScalarIScalarO(connect, kFree, 2, 0, kNilOptions, id2);
    assert(kIOReturnSuccess == status);
    printf("free ID: %d\n", id2);

    exit(0);    
}


