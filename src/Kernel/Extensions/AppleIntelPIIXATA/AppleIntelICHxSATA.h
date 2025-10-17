/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
#ifndef _APPLEINTELICHXSATA_H
#define _APPLEINTELICHXSATA_H

#include "AppleIntelPIIXPATA.h"
#ifndef kIOPolledInterfaceSupportKey
#include <IOKit/IOPolledInterface.h>
#endif

class AppleIntelICHxSATA : public AppleIntelPIIXPATA
{
    OSDeclareDefaultStructors( AppleIntelICHxSATA )

    class AppleIntelICHxSATAPolledAdapter* polledAdapter;

protected:
    bool             _initPortEnable;

    virtual IOReturn selectDevice( ataUnitID unit );

    //override for polling
    virtual void executeEventCallouts(  ataEventCode event, ataUnitID unit);
    virtual IOReturn startTimer( UInt32 inMS);
    virtual void stopTimer( void );

public:
    virtual bool     start( IOService * provider );

    virtual IOReturn provideBusInfo( IOATABusInfo * infoOut );
    
    virtual UInt32   scanForDrives( void );

    virtual IOReturn setPowerState( unsigned long stateIndex,
                                    IOService *   whatDevice );

public:
    virtual void pollEntry( void );
    virtual void transitionFixup( void );
};

class AppleIntelICHxSATAPolledAdapter : public IOPolledInterface

{
    OSDeclareDefaultStructors(AppleIntelICHxSATAPolledAdapter)

protected:
    AppleIntelICHxSATA* owner;
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
    
    void setOwner( AppleIntelICHxSATA* owner );
};

#endif /* !_APPLEINTELICHXSATA_H */
