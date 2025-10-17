/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#ifndef _APPLEINTELPIIXATAROOT_H
#define _APPLEINTELPIIXATAROOT_H

#include <IOKit/IOLocks.h>
#include <IOKit/pci/IOPCIDevice.h>

class AppleIntelPIIXATARoot : public IOService
{
    OSDeclareDefaultStructors( AppleIntelPIIXATARoot )

protected:
    OSSet *       _nubs;
    OSSet *       _openNubs;
    IOPCIDevice * _provider;
    IOLock *      _pciConfigLock;

    virtual OSSet * createATAChannelNubs( void );

    virtual OSDictionary * createNativeModeChannelInfo( UInt32 ataChannel,
                                                        UInt32 channelMode );

    virtual OSDictionary * createLegacyModeChannelInfo( UInt32 ataChannel,
                                                        UInt32 channelMode );

    virtual OSDictionary * createChannelInfo( UInt32 ataChannel,
                                              UInt32 channelMode,
                                              UInt16 commandPort,
                                              UInt16 controlPort,
                                              UInt8  interruptVector );

    virtual IORegistryEntry * getDTChannelEntry( int channelID );

public:
    virtual IOService * probe( IOService * provider,
                               SInt32 *    score );

    virtual bool start( IOService * provider );

    virtual void free( void );

    virtual bool handleOpen( IOService *  client,
                             IOOptionBits options,
                             void *       arg );
    
    virtual void handleClose( IOService *  client,
                              IOOptionBits options );

    virtual bool handleIsOpen( const IOService * client ) const;

    virtual void pciConfigWrite8( UInt8 offset,
                                  UInt8 data,
                                  UInt8 mask = 0xff );
    
    virtual void pciConfigWrite16( UInt8  offset,
                                   UInt16 data,
                                   UInt16 mask = 0xffff );

	virtual void setSerialATAPortEnable( UInt32 port, bool enable );

	virtual bool getSerialATAPortPresentStatus( UInt32 port );

    virtual bool serializeProperties( OSSerialize * s ) const;
};

#endif /* !_APPLEINTELPIIXATAROOT_H */
