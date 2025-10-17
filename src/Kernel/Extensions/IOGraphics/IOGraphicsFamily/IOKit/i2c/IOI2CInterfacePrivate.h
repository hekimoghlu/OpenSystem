/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
#ifndef _IOKIT_IOI2CINTERFACEPRIVATE_H
#define _IOKIT_IOI2CINTERFACEPRIVATE_H

#ifdef KERNEL
#include <IOKit/IOLib.h>
#include <IOKit/IOUserClient.h>

#include <IOKit/ndrvsupport/IONDRVFramebuffer.h>
#endif /* KERNEL */

#include <IOKit/i2c/IOI2CInterface.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

enum { kIOI2CInlineBufferBytes = 1024 };

struct IOI2CBuffer
{
    IOI2CRequest        request;
    UInt8               inlineBuffer[ kIOI2CInlineBufferBytes ];
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifdef KERNEL

#pragma pack(push, 4)

struct IOI2CRequest_10_5_0
{
    UInt64              __reservedA;
    IOReturn            result;
    uint32_t            completion;
    IOOptionBits        commFlags;
    uint64_t            minReplyDelay;
    uint8_t             sendAddress;
    uint8_t             sendSubAddress;
    uint8_t             __reservedB[2];
    IOOptionBits        sendTransactionType;
    uint32_t            sendBuffer;
    uint32_t            sendBytes;
    uint8_t             replyAddress;
    uint8_t             replySubAddress;
    uint8_t             __reservedC[2];
    IOOptionBits        replyTransactionType;
    uint32_t            replyBuffer;
    uint32_t            replyBytes;
    uint32_t            __reservedD[16];
};

#pragma pack(pop)

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

class IOI2CInterfaceUserClient : public IOUserClient
{
    OSDeclareDefaultStructors(IOI2CInterfaceUserClient)

protected:
    task_t      fTask;

public:
    // IOUserClient methods
    virtual IOReturn clientClose( void ) APPLE_KEXT_OVERRIDE;

    virtual IOService * getService( void ) APPLE_KEXT_OVERRIDE;

    virtual IOExternalMethod * getTargetAndMethodForIndex(
                                        IOService ** targetP, UInt32 index ) APPLE_KEXT_OVERRIDE;
    static IOI2CInterfaceUserClient * withTask( task_t owningTask );
    virtual bool start( IOService * provider ) APPLE_KEXT_OVERRIDE;
    virtual IOReturn setProperties( OSObject * properties ) APPLE_KEXT_OVERRIDE;

    virtual bool willTerminate(IOService *provider, IOOptionBits options) APPLE_KEXT_OVERRIDE;
    virtual bool didTerminate(IOService *provider, IOOptionBits options, bool *defer) APPLE_KEXT_OVERRIDE;
    virtual bool requestTerminate(IOService *provider, IOOptionBits options) APPLE_KEXT_OVERRIDE;
    virtual bool terminate(IOOptionBits options = 0) APPLE_KEXT_OVERRIDE;
    virtual bool finalize(IOOptionBits options) APPLE_KEXT_OVERRIDE;
    virtual void stop(IOService *provider) APPLE_KEXT_OVERRIDE;
    virtual void free() APPLE_KEXT_OVERRIDE;

    // others

    virtual IOReturn extAcquireBus( void );
    virtual IOReturn extReleaseBus( void );
    virtual IOReturn extIO( void * inStruct, void * outStruct,
                            IOByteCount inSize, IOByteCount * outSize );
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#endif /* KERNEL */

#endif /* ! _IOKIT_IOI2CINTERFACEPRIVATE_H */

