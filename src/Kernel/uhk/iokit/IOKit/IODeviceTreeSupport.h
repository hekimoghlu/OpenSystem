/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
 * Copyright (c) 1998 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */

#ifndef _IOKIT_IODEVICETREE_H
#define _IOKIT_IODEVICETREE_H

#include <IOKit/IORegistryEntry.h>
#include <libkern/c++/OSData.h>
#include <libkern/c++/OSPtr.h>

class IODeviceMemory;
class IOService;

extern const IORegistryPlane *  gIODTPlane;

extern const OSSymbol *         gIODTPHandleKey;

extern const OSSymbol *         gIODTCompatibleKey;
extern const OSSymbol *         gIODTTypeKey;
extern const OSSymbol *         gIODTModelKey;
extern const OSSymbol *         gIODTBridgeModelKey;
extern const OSSymbol *         gIODTTargetTypeKey;

extern const OSSymbol *         gIODTAAPLInterruptsKey;
extern const OSSymbol *         gIODTDefaultInterruptController;
extern const OSSymbol *         gIODTNWInterruptMappingKey;

extern const OSData *           gIODTAssociatedServiceKey;
#define kIODTAssociatedServiceKey       "associated-service"

LIBKERN_RETURNS_NOT_RETAINED IORegistryEntry * IODeviceTreeAlloc( void * dtTop );


bool IODTMatchNubWithKeys( IORegistryEntry * nub,
    const char * keys );

bool IODTCompareNubName( const IORegistryEntry * regEntry,
    OSString * name,
    LIBKERN_RETURNS_RETAINED_ON_NONZERO OSString ** matchingName );
bool IODTCompareNubName( const IORegistryEntry * regEntry,
    OSString * name,
    OSSharedPtr<OSString>& matchingName );

enum {
	kIODTRecursive       = 0x00000001,
	kIODTExclusive       = 0x00000002,
};

LIBKERN_RETURNS_RETAINED OSCollectionIterator * IODTFindMatchingEntries( IORegistryEntry * from,
    IOOptionBits options, const char * keys );

#if !defined(__arm64__)
typedef SInt32 (*IODTCompareAddressCellFunc)
(UInt32 cellCount, UInt32 left[], UInt32 right[]);
#else
typedef SInt64 (*IODTCompareAddressCellFunc)
(UInt32 cellCount, UInt32 left[], UInt32 right[]);
#endif

typedef void (*IODTNVLocationFunc)
(IORegistryEntry * entry,
    UInt8 * busNum, UInt8 * deviceNum, UInt8 * functionNum );

void IODTSetResolving( IORegistryEntry *        regEntry,
    IODTCompareAddressCellFunc      compareFunc,
    IODTNVLocationFunc              locationFunc );

void IODTGetCellCounts( IORegistryEntry * regEntry,
    UInt32 * sizeCount, UInt32 * addressCount);

bool IODTResolveAddressCell( IORegistryEntry * regEntry,
    UInt32 cellsIn[],
    IOPhysicalAddress * phys, IOPhysicalLength * len );

LIBKERN_RETURNS_NOT_RETAINED OSArray *
IODTResolveAddressing( IORegistryEntry * regEntry,
    const char * addressPropertyName,
    IODeviceMemory * parent );

struct IONVRAMDescriptor {
	unsigned int format:4;
	unsigned int marker:1;
	unsigned int bridgeCount:3;
	unsigned int busNum:2;
	unsigned int bridgeDevices:6 * 5;
	unsigned int functionNum:3;
	unsigned int deviceNum:5;
} __attribute__((aligned(2), packed));

IOReturn IODTMakeNVDescriptor( IORegistryEntry * regEntry,
    IONVRAMDescriptor * hdr );

LIBKERN_RETURNS_NOT_RETAINED OSData *
IODTFindSlotName( IORegistryEntry * regEntry, UInt32 deviceNumber );

const OSSymbol * IODTInterruptControllerName(
	IORegistryEntry * regEntry );

bool IODTMapInterrupts( IORegistryEntry * regEntry );

enum {
	kIODTInterruptShared = 0x00000001
};
IOReturn IODTGetInterruptOptions( IORegistryEntry * regEntry, int source, IOOptionBits * options );

#ifdef __cplusplus
extern "C" {
#endif

IOReturn IONDRVLibrariesInitialize( IOService * provider );

#ifdef __cplusplus
}
#endif

#endif /* _IOKIT_IODEVICETREE_H */
