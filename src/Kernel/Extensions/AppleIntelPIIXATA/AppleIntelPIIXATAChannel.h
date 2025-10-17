/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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
#ifndef _APPLEINTELPIIXATACHANNEL_H
#define _APPLEINTELPIIXATACHANNEL_H

#include <IOKit/IOService.h>

class AppleIntelPIIXATAChannel : public IOService
{
    OSDeclareDefaultStructors( AppleIntelPIIXATAChannel )

protected:
    IOService *    _provider;
    UInt16         _cmdBlock;
    UInt16         _ctrBlock;
    UInt8          _irq;
    UInt8          _pioModeMask;
    UInt8          _dmaModeMask;
    UInt8          _udmaModeMask;
    UInt32         _channelNum;
    UInt32         _channelMode;
    const char *   _controllerName;
    bool           _hasSharedDriveTimings;

    virtual bool   getNumberValue( const char * propKey,
                                   void       * outValue,
                                   UInt32       outBits );

    virtual bool   setupInterrupt( IOService * provider, UInt32 line );

    virtual void   mergeProperties( OSDictionary * properties );

public:
    virtual bool   init( IOService *       provider,
                         OSDictionary *    properties,
                         IORegistryEntry * dtEntry = 0 );

    virtual bool   matchPropertyTable( OSDictionary * table,
                                       SInt32 *       score );

    virtual UInt16 getCommandBlockAddress( void ) const;

    virtual UInt16 getControlBlockAddress( void ) const;

    virtual UInt8  getInterruptVector( void ) const;

    virtual UInt8  getPIOModeMask( void ) const;

    virtual UInt8  getDMAModeMask( void ) const;

    virtual UInt8  getUltraDMAModeMask( void ) const;

    virtual UInt32 getChannelNumber( void ) const;

    virtual UInt32 getChannelMode( void ) const;

    virtual bool   hasSharedDriveTimings( void ) const;

    virtual const char * getControllerName( void ) const;

    virtual UInt32 getMaxDriveUnits( void ) const;

    virtual UInt32 getSerialATAPortForDrive( UInt32 unit ) const;

    virtual void   setSerialATAPortEnableForDrive( UInt32 unit, bool enable );

    virtual bool   getSerialATAPortPresentStatusForDrive( UInt32 unit );

    virtual bool   handleOpen( IOService *  client,
                               IOOptionBits options,
                               void *       arg );

    virtual void   handleClose( IOService *  client,
                                IOOptionBits options );

    virtual void   pciConfigWrite8( UInt8 offset,
                                    UInt8 data,
                                    UInt8 mask = 0xff );

    virtual void   pciConfigWrite16( UInt8  offset,
                                     UInt16 data,
                                     UInt16 mask = 0xffff );
};

#endif /* !_APPLEINTELPIIXATACHANNEL_H */
