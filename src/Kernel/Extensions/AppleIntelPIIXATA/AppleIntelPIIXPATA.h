/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
#ifndef _APPLEINTELPIIXPATA_H
#define _APPLEINTELPIIXPATA_H

#include <IOKit/IOFilterInterruptEventSource.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/pci/IOPCIDevice.h>
#include <IOKit/ata/IOATATypes.h>
#include <IOKit/ata/IOPCIATA.h>
#include <IOKit/ata/IOATAController.h>
#include <IOKit/ata/ATADeviceNub.h>
#include <IOKit/IOPolledInterface.h>
#include "AppleIntelPIIXATATiming.h"
#include "AppleIntelPIIXATAHW.h"
#include "AppleIntelPIIXATAKeys.h"
#include "AppleIntelPIIXATAChannel.h"
#include <IOKit/acpi/IOACPIPlatformDevice.h>

class AppleIntelPIIXPATA : public IOPCIATA
{
    OSDeclareDefaultStructors( AppleIntelPIIXPATA )

protected:
    
    static const UInt32 kMaxDrives = 2;
    class AppleIntelICHxPATAPolledAdapter* polledPATAAdapter;
    /*
     * General and PIIX specific ATA controller properties.
     */
    UInt16                        _cmdBlock;
    UInt16                        _ctrBlock;
    UInt16                        _ioBMOffset;
    UInt32                        _channel;

    /*
     * References to parent(s), and other objects.
     */
    IOPCIDevice *                 _pciDevice;
    IOInterruptEventSource *      _intSrc;
    AppleIntelPIIXATAChannel *    _provider;

    /*
     * Currently selected timings for each drive.
     */
    const PIIXTiming *            _pioTiming[  kMaxDrives ];
    const PIIXTiming *            _dmaTiming[  kMaxDrives ];
    const PIIXUDMATiming *        _udmaTiming[ kMaxDrives ];

    /*
     * The timing register values that correspond to the
     * selected timings.
     */
    UInt16                        _idetim[ kMaxDrives ];
    UInt8                         _sidetim;
    UInt8                         _udmactl;
    UInt16                        _udmatim;
    UInt16                        _ideConfig;
    bool                          _initTimingRegisters;

	IOACPIPlatformDevice*         	_pciACPIDevice;
	bool							_drivePowerOn;
	IONotifier*						_interestNotifier;
	IOBufferMemoryDescriptor*		_prdBuffer;
	
    /* Interrupt event source action */
    
    static void interruptOccurred( OSObject *               owner,
                                   IOInterruptEventSource * src,
                                   int                      count );

    /* Interrupt event source filter */

    static bool interruptFilter( OSObject * owner,
                                 IOFilterInterruptEventSource * src );

    /*
     * PIIX support functions.
     */
    virtual bool configurePCIDevice( IOPCIDevice * device,
                                     UInt32        channel );

    virtual bool getBMBaseAddress( IOPCIDevice * provider,
                                   UInt32        channel,
                                   UInt16 *      addrOut );

    virtual void resetTimingsForDevice( ataUnitID unit );

    virtual void writeTimingRegisters( ataUnitID unit = kATADevice0DeviceID );

    virtual void computeUDMATimingRegisters( ataUnitID unit );

    virtual void computeTimingRegisters( ataUnitID unit );

    virtual void selectIOTiming( ataUnitID unit );

    virtual bool setDriveProperty( UInt32       driveUnit,
                                   const char * key,
                                   UInt32       value,
                                   UInt32       numberOfBits);

    virtual IOReturn synchronousIO( void );

    virtual void initForPM( IOService * provider );
	
	// acpi media notify
	IOACPIPlatformDevice* getACPIParent( void );
	bool hasMediaNotify(IOACPIPlatformDevice* acpi_device);
	void turnOffDrive( void );
	void turnOnDrive( void );
	
	static IOReturn mediaInterestHandler( void* target, 
										void* refCon,
										UInt32 messageType, 
										IOService* provider,
										void* messageArgument,
										vm_size_t argSize);
	
	IOReturn handleInsert( void );									
																												
	// override from IOATAController
	virtual void completeIO( IOReturn commandResult);
	virtual IOReturn dispatchNext( void );
	
	virtual IOATAController::transState	determineATAPIState(void);
	
	//override for polling
    virtual void executeEventCallouts(  ataEventCode event, ataUnitID unit);
    virtual IOReturn startTimer( UInt32 inMS);
    virtual void stopTimer( void );

	
public:
    virtual void pollEntry( void );
    virtual void transitionFixup( void );


public:
    /* IOService overrides */

    virtual bool start( IOService * provider );

    virtual void stop( IOService * provider );

    virtual void free( void );

    virtual IOWorkLoop * getWorkLoop( void ) const;

    virtual IOReturn message( UInt32      type,
                              IOService * provider,
                              void *      argument );

    virtual IOReturn setPowerState( unsigned long stateIndex,
                                    IOService *   whatDevice );

    /* Mandatory IOATAController overrides */

    virtual bool configureTFPointers( void );

    virtual IOReturn provideBusInfo( IOATABusInfo * infoOut );

    virtual IOReturn getConfig( IOATADevConfig * configOut,
                                UInt32           unit );

    virtual IOReturn selectConfig( IOATADevConfig * config,
                                   UInt32           unit );

    /* Optional IOATAController overrides */

    virtual UInt32 scanForDrives( void );

    virtual IOReturn handleQueueFlush( void );

    /* Optional IOPCIATA overrides to support large transfers */

    virtual bool allocDMAChannel( void );

    virtual void initATADMAChains( PRD * descPtr );

    virtual IOReturn createChannelCommands( void );

    virtual bool freeDMAChannel( void );
	
    /* Optional IOPCIATA override to prevent spurius interrupts */
	
	virtual IOReturn selectDevice( ataUnitID unit );

};

class AppleIntelICHxPATAPolledAdapter : public IOPolledInterface

{
    OSDeclareDefaultStructors(AppleIntelICHxPATAPolledAdapter)

protected:
    AppleIntelPIIXPATA* owner;
    bool pollingActive;

public:
    virtual IOReturn probe(IOService * target);
    virtual IOReturn open( IOOptionBits state, IOMemoryDescriptor * buffer);
    virtual IOReturn close(IOOptionBits state);

    virtual IOReturn startIO(uint32_t 	        operation,
                             uint32_t           bufferOffset,
                             uint64_t	        deviceOffset,
                             uint64_t	        length,
                             IOPolledCompletion completion) ;

    virtual IOReturn checkForWork(void);
	
    bool isPolling( void );
    
    void setOwner( AppleIntelPIIXPATA* owner );
};


#endif /* !_APPLEINTELPIIXPATA_H */
