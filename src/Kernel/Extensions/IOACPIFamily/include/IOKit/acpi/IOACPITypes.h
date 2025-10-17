/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
#ifndef __IOKIT_IOACPITYPES_H
#define __IOKIT_IOACPITYPES_H

#include <IOKit/IOMessage.h>

extern const IORegistryPlane * gIOACPIPlane;
extern const OSSymbol *        gIOACPIHardwareIDKey;
extern const OSSymbol *        gIOACPIUniqueIDKey;
extern const OSSymbol *        gIOACPIAddressKey;
extern const OSSymbol *        gIOACPIDeviceStatusKey;

#pragma pack(1)

struct IOACPIAddressSpaceDescriptor {
    UInt32  resourceType;
    UInt32  generalFlags;
    UInt32  typeSpecificFlags;
    UInt32  reserved1;
    UInt64  granularity;
    UInt64  minAddressRange;
    UInt64  maxAddressRange;
    UInt64  translationOffset;
    UInt64  addressLength;
    UInt64  reserved2;
    UInt64  reserved3;
    UInt64  reserved4;
};

enum {
    kIOACPIMemoryRange    = 0,
    kIOACPIIORange        = 1,
    kIOACPIBusNumberRange = 2
};

typedef UInt32 IOACPIAddressSpaceID;

enum {
    kIOACPIAddressSpaceIDSystemMemory       = 0,
    kIOACPIAddressSpaceIDSystemIO           = 1,
    kIOACPIAddressSpaceIDPCIConfiguration   = 2,
    kIOACPIAddressSpaceIDEmbeddedController = 3,
    kIOACPIAddressSpaceIDSMBus              = 4
};

/*
 * Address space operation
 */
enum {
    kIOACPIAddressSpaceOpRead  = 0,
    kIOACPIAddressSpaceOpWrite = 1
};

/*
 * 64-bit ACPI address
 */
union IOACPIAddress {
    UInt64 addr64;
    struct {
        unsigned int offset     :16;
        unsigned int function   :3;
        unsigned int device     :5;
        unsigned int bus        :8;
        unsigned int segment    :16;
        unsigned int reserved   :16;
    } pci;
};

/*
 * Address space handler
 */
typedef IOReturn (*IOACPIAddressSpaceHandler)( UInt32         operation,
                                               IOACPIAddress  address,
                                               UInt64 *       value,
                                               UInt32         bitWidth,
                                               UInt32         bitOffset,
                                               void *         context );

/*
 * ACPI fixed event types
 */
enum {
    kIOACPIFixedEventPMTimer       = 0,
    kIOACPIFixedEventPowerButton   = 2,
    kIOACPIFixedEventSleepButton   = 3,
    kIOACPIFixedEventRealTimeClock = 4
};

#pragma pack()

/*
 * FIXME: Move to xnu/iokit to reserve the ACPI family code.
 */
#ifndef sub_iokit_acpi
#define sub_iokit_acpi   err_sub(10)
#endif

/*
 * ACPI notify message sent to all clients and interested parties.
 * The notify code can be read from the argument as an UInt32.
 */
#define kIOACPIMessageDeviceNotification  iokit_family_msg(sub_iokit_acpi, 0x10)

/*
 * ACPI device power states
 */
enum {
    kIOACPIDevicePowerStateD0    = 0,
    kIOACPIDevicePowerStateD1    = 1,
    kIOACPIDevicePowerStateD2    = 2,
    kIOACPIDevicePowerStateD3    = 3,
    kIOACPIDevicePowerStateCount = 4
};

#endif /* !__IOKIT_IOACPITYPES_H */
