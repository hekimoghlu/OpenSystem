/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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


#include <IOKit/IOUserClient.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/pci/IOPCIDevice.h>

#if IOMEMORYDESCRIPTOR_SUPPORTS_DMACOMMAND
#include <IOKit/IODMACommand.h>
#endif

#include "AppleSamplePCIShared.h"

class AppleSamplePCI : public IOService
{
    /*
     * Declare the metaclass information that is used for runtime
     * typechecking of IOKit objects.
     */

    OSDeclareDefaultStructors( AppleSamplePCI );

private:
    IOPCIDevice *        fPCIDevice;
    IOMemoryDescriptor * fLowMemory;

public:
    /* IOService overrides */
    virtual bool start( IOService * provider );
    virtual void stop( IOService * provider );
    /* Other methods */
    IOMemoryDescriptor * copyGlobalMemory( void );
    IOReturn generateDMAAddresses( IOMemoryDescriptor * memDesc );
};

class AppleSamplePCIUserClient : public IOUserClient
{
    /*
     * Declare the metaclass information that is used for runtime
     * typechecking of IOKit objects.
     */

    OSDeclareDefaultStructors( AppleSamplePCIUserClient );

private:
    AppleSamplePCI *            fDriver;
    IOBufferMemoryDescriptor *  fClientSharedMemory;
    AppleSampleSharedMemory *   fClientShared;
    task_t                      fTask;
    SInt32                      fOpenCount;

public:
    /* IOService overrides */
    virtual bool start( IOService * provider );
    virtual void stop( IOService * provider );

    /* IOUserClient overrides */
    virtual bool initWithTask( task_t owningTask, void * securityID,
                                                UInt32 type,  OSDictionary * properties );
    virtual IOReturn clientClose( void );

    virtual IOExternalMethod * getTargetAndMethodForIndex(
                                            IOService ** targetP, UInt32 index );

    virtual IOReturn externalMethod( uint32_t selector, IOExternalMethodArguments * arguments,
                                        IOExternalMethodDispatch * dispatch = 0, OSObject * target = 0, void * reference = 0 );


    virtual IOReturn clientMemoryForType( UInt32 type,
                                            IOOptionBits * options,
                                            IOMemoryDescriptor ** memory );
    /* External methods */
    virtual IOReturn method1( UInt32 * dataIn, UInt32 * dataOut,
                                                IOByteCount inputCount, IOByteCount * outputCount );
    virtual IOReturn method2( AppleSampleStructForMethod2 * structIn, 
                                            AppleSampleResultsForMethod2 * structOut,
                                            IOByteCount inputSize, IOByteCount * outputSize );
};

