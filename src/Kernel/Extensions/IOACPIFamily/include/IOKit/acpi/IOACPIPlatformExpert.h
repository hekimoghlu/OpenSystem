/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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
#ifndef _IOKIT_IOACPIPLATFORMEXPERT_H
#define _IOKIT_IOACPIPLATFORMEXPERT_H

#include <IOKit/IOPlatformExpert.h>           // superclass
#include <IOKit/acpi/IOACPIPlatformDevice.h>  // children

class IOACPIPlatformExpert : public IODTPlatformExpert
{
    OSDeclareAbstractStructors( IOACPIPlatformExpert )

    friend class IOACPIPlatformDevice;

protected:
    /*! @struct ExpansionData
        @discussion This structure will be used to expand the capablilties
                    of the class in the future.
     */
    struct ExpansionData { };

    /*! @var reserved
        Reserved for future use. (Internal use only)
     */
    ExpansionData *  reserved;

public:
    virtual bool     start( IOService * provider );

protected:
    // Map ACPI event to interrupt event source index

    virtual SInt32   installDeviceInterruptForFixedEvent(
                                  IOService *  device,
                                  UInt32       fixedEvent ) = 0;

    virtual SInt32   installDeviceInterruptForGPE(
                                  IOService *  device,
                                  UInt32       gpeNumber,
                                  void *       gpeBlockDevice,
                                  IOOptionBits options ) = 0;

    // ACPI global lock acquisition

    virtual IOReturn acquireGlobalLock( IOService *             client,
                                        UInt32 *                lockToken,
                                        const mach_timespec_t * timeout ) = 0;

    virtual void     releaseGlobalLock( IOService * client,
                                        UInt32      lockToken ) = 0;

    // ACPI method and object evaluation

    virtual IOReturn validateObject( IOACPIPlatformDevice * device,
                                     const OSSymbol *       objectName ) = 0;

    virtual IOReturn validateObject( IOACPIPlatformDevice * device,
                                     const char *           objectName );

    virtual IOReturn evaluateObject( IOACPIPlatformDevice * device,
                                     const OSSymbol *       objectName,
                                     OSObject **            result,
                                     OSObject *             params[],
                                     IOItemCount            paramCount,
                                     IOOptionBits           options ) = 0;

    virtual IOReturn evaluateObject( IOACPIPlatformDevice * device,
                                     const char *           objectName,
                                     OSObject **            result,
                                     OSObject *             params[],
                                     IOItemCount            paramCount,
                                     IOOptionBits           options );

    // ACPI table

    virtual const OSData * getACPITableData(
                                     const char * tableName,
                                     UInt32       tableInstance ) = 0;

    // Address space handler

    virtual IOReturn registerAddressSpaceHandler(
                                   IOACPIPlatformDevice *    device,
                                   IOACPIAddressSpaceID      spaceID,
                                   IOACPIAddressSpaceHandler handler,
                                   void *                    context,
                                   IOOptionBits              options ) = 0;

    virtual void     unregisterAddressSpaceHandler(
                                   IOACPIPlatformDevice *    device,
                                   IOACPIAddressSpaceID      spaceID,
                                   IOACPIAddressSpaceHandler handler,
                                   IOOptionBits              options ) = 0;

    // Address space read/write

    virtual IOReturn readAddressSpace(  UInt64 *             value,
                                        IOACPIAddressSpaceID spaceID,
                                        IOACPIAddress        address,
                                        UInt32               bitWidth,
                                        UInt32               bitOffset,
                                        IOOptionBits         options ) = 0;

    virtual IOReturn writeAddressSpace( UInt64               value,
                                        IOACPIAddressSpaceID spaceID,
                                        IOACPIAddress        address,
                                        UInt32               bitWidth,
                                        UInt32               bitOffset,            
                                        IOOptionBits         options ) = 0;

    // Device power management

    virtual IOReturn setDevicePowerState( IOACPIPlatformDevice * device,
                                          UInt32 powerState ) = 0;

    virtual IOReturn getDevicePowerState( IOACPIPlatformDevice * device,
                                          UInt32 * powerState ) = 0;

    virtual IOReturn setDeviceWakeEnable( IOACPIPlatformDevice * device,
                                          bool enable ) = 0;

    // vtable padding

    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  0 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  1 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  2 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  3 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  4 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  5 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  6 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  7 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  8 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert,  9 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 10 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 11 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 12 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 13 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 14 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 15 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 16 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 17 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 18 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 19 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 20 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 21 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 22 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 23 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 24 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 25 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 26 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 27 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 28 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 29 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 30 );
    OSMetaClassDeclareReservedUnused( IOACPIPlatformExpert, 31 );
};

#endif /* !_IOKIT_IOACPIPLATFORMEXPERT_H */
