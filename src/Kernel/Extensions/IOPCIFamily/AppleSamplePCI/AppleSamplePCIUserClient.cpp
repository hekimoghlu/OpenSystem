/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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
#include "AppleSamplePCI.h"
#include <IOKit/IOLib.h>
#include <IOKit/assert.h>

/* 
 * Define the metaclass information that is used for runtime
 * typechecking of IOKit objects. We're a subclass of IOUserClient.
 */

#define super IOUserClient
OSDefineMetaClassAndStructors( AppleSamplePCIUserClient, IOUserClient );

/* 
 * Since this sample uses the IOUserClientClass property, the AppleSamplePCIUserClient
 * is created automatically in response to IOServiceOpen(). More complex applications
 * might have several kinds of clients each with a different IOUserClient subclass,
 * with different enumerated types. In that case the AppleSamplePCI class must implement
 * the newUserClient() method (see IOService.h headerdoc).
 */

bool AppleSamplePCIUserClient::initWithTask( task_t owningTask, void * securityID,
                                             UInt32 type,  OSDictionary * properties )
{
    IOLog("AppleSamplePCIUserClient::initWithTask(type %ld)\n", type);
    
    fTask = owningTask;

    return( super::initWithTask( owningTask, securityID, type, properties ));
}

bool AppleSamplePCIUserClient::start( IOService * provider )
{
    IOLog("AppleSamplePCIUserClient::start\n");

    if( !super::start( provider ))
        return( false );

    /*
     * Our provider is the AppleSamplePCI object.
     */

    assert( OSDynamicCast( AppleSamplePCI, provider ));
    fDriver = (AppleSamplePCI *) provider;

    /*
     * Set up some memory to be shared between this user client instance and its
     * client process. The client will call in to map this memory, and iokit
     * will call clientMemoryForType to obtain this memory descriptor.
     */

    fClientSharedMemory = IOBufferMemoryDescriptor::withOptions(
                kIOMemoryKernelUserShared, sizeof( AppleSampleSharedMemory ));
    if( !fClientSharedMemory)
        return( false );

    fClientShared = (AppleSampleSharedMemory *) fClientSharedMemory->getBytesNoCopy();
    fClientShared->field1 = 0x11111111;
    fClientShared->field2 = 0x22222222;
    fClientShared->field3 = 0x33333333;
    strcpy( fClientShared->string, "some data" );
    fOpenCount = 1;

    return( true );
}


/*
 * Kill ourselves off if the client closes its connection or the client dies.
 */

IOReturn AppleSamplePCIUserClient::clientClose( void )
{
    if( !isInactive())
        terminate();

    return( kIOReturnSuccess );
}

/* 
 * stop will be called during the termination process, and should free all resources
 * associated with this client.
 */
void AppleSamplePCIUserClient::stop( IOService * provider )
{
    IOLog("AppleSamplePCIUserClient::stop\n");

    if( fClientSharedMemory) {
        fClientSharedMemory->release();
        fClientShared = 0;
    }

    super::stop( provider );
}

/*
 * Lookup the external methods - supply a description of the parameters 
 * available to be called 
 */

IOExternalMethod * AppleSamplePCIUserClient::getTargetAndMethodForIndex(
                                                    IOService ** targetP, UInt32 index )
{
    static const IOExternalMethod methodDescs[kAppleSampleNumMethods] = {

      { NULL, (IOMethod) &AppleSamplePCIUserClient::method1,
        kIOUCStructIStructO, kIOUCVariableStructureSize, kIOUCVariableStructureSize },

      { NULL, (IOMethod) &AppleSamplePCIUserClient::method2,
        kIOUCStructIStructO, sizeof(AppleSampleStructForMethod2), sizeof(AppleSampleResultsForMethod2) },
    };

    *targetP = this;
    if( index < kAppleSampleNumMethods)
        return( (IOExternalMethod *)(methodDescs + index) );
    else
        return NULL;
}

IOReturn AppleSamplePCIUserClient::externalMethod(
        uint32_t selector, IOExternalMethodArguments * arguments,
        IOExternalMethodDispatch * dispatch, OSObject * target, void * reference )
{

    return (super::externalMethod(selector, arguments, NULL, this, NULL));

    IOReturn err;

    switch (selector)
    {
        case kAppleSampleMethod1:
            err = method1( (UInt32 *) arguments->structureInput, 
                            (UInt32 *)  arguments->structureOutput,
                            arguments->structureInputSize, (IOByteCount *) &arguments->structureOutputSize );
            break;

        case kAppleSampleMethod2:
            err = method2( (AppleSampleStructForMethod2 *) arguments->structureInput, 
                            (AppleSampleResultsForMethod2 *)  arguments->structureOutput,
                            arguments->structureInputSize, (IOByteCount *) &arguments->structureOutputSize );
            break;

        default:
            err = kIOReturnBadArgument;
            break;
    }

    IOLog("externalMethod(%d) 0x%x", selector, err);

    return (err);
}

/*
 * Implement each of the external methods described above.
 */

IOReturn AppleSamplePCIUserClient::method1(
                                           UInt32 * dataIn, UInt32 * dataOut,
                                           IOByteCount inputSize, IOByteCount * outputSize )
{
    IOReturn    ret;
    IOItemCount count;

    IOLog("AppleSamplePCIUserClient::method1(");

    if( *outputSize < inputSize)
        return( kIOReturnNoSpace );

    count = inputSize / sizeof( UInt32 );
    for( UInt32 i = 0; i < count; i++ ) {
        IOLog("%08lx, ", dataIn[i]);
        dataOut[i] = dataIn[i] ^ 0xffffffff;
    }

    ret = kIOReturnSuccess;
    IOLog(")\n");
    *outputSize = count * sizeof( UInt32 );

    return( ret );
}

IOReturn AppleSamplePCIUserClient::method2( AppleSampleStructForMethod2 * structIn, 
                                            AppleSampleResultsForMethod2 * structOut,
                                            IOByteCount inputSize, IOByteCount * outputSize )

{
    IOReturn err;
    IOMemoryDescriptor * memDesc = 0;
    UInt32 param1 = structIn->parameter1;
    mach_vm_address_t clientAddr = structIn->data_pointer;
    mach_vm_size_t size = structIn->data_length;

    IOLog("AppleSamplePCIUserClient::method2(%lx)\n", param1);

    IOLog( "fClientShared->string == \"%s\"\n", fClientShared->string );

    structOut->results1 = 0x87654321;

    do
    {
        // construct a memory descriptor for the out of line client memory

        // old 32 bit API - this will fail and log a backtrace if the task is 64 bit
        memDesc = IOMemoryDescriptor::withAddress( clientAddr, size, kIODirectionNone, fTask );
        if( !memDesc) {
            IOLog("IOMemoryDescriptor::withAddress failed\n");
        } else {
            memDesc->release();
        }

        // 64 bit API - works on all tasks, whether 64 bit or 32 bit
        memDesc = IOMemoryDescriptor::withAddressRange( clientAddr, size, kIODirectionNone, fTask );
        if( !memDesc) {
            IOLog("IOMemoryDescriptor::withAddress failed\n");
            err = kIOReturnVMError;
            continue;
        }

        // wire it and make sure we can write it
        err = memDesc->prepare( kIODirectionOutIn );
        if( kIOReturnSuccess != err) {
            IOLog("IOMemoryDescriptor::prepare failed(%x)\n", err);
            continue;
        }

        // Generate a DMA list for the client memory
        err = fDriver->generateDMAAddresses(memDesc);

        // Other methods to access client memory:

        // readBytes/writeBytes allow programmed I/O to/from an offset in the buffer
        char pioBuffer[ 200 ];
        memDesc->readBytes(32, &pioBuffer, sizeof( pioBuffer));
        IOLog("readBytes: \"%s\"\n", pioBuffer);

        // map() will create a mapping in the kernel address space.
        IOMemoryMap * memMap = memDesc->map();
        if( memMap) {
            char * address = (char *) memMap->getVirtualAddress();
            IOLog("kernel mapped: \"%s\"\n", address + 32);
            memMap->release();
        } else
            IOLog("memDesc map(kernel) failed\n");

        // this map() will create a mapping in the users (the client of this IOUserClient) address space.
        memMap = memDesc->map(fTask, 0, kIOMapAnywhere);
        if( memMap)
        {
            // old 32 bit API - this will truncate and log a backtrace if the task is 64 bit
            IOVirtualAddress address32 = memMap->getVirtualAddress();
            IOLog("user32 mapped: 0x%x\n", address32);

            // new 64 bit API - same for 32 bit and 64 bit client tasks
            mach_vm_address_t address64 = memMap->getAddress();
            IOLog("user64 mapped: 0x%qx\n", address64);
            memMap->release();
        } else
            IOLog("memDesc map(user) failed\n");

        // Done with the I/O now.
        memDesc->complete( kIODirectionOutIn );

    } while( false );

    if( memDesc)
        memDesc->release();

    return( err );
}

/*
 * Shared memory support. Supply a IOMemoryDescriptor instance to describe
 * each of the kinds of shared memory available to be mapped into the client
 * process with this user client.
 */

IOReturn AppleSamplePCIUserClient::clientMemoryForType(
                                UInt32 type,
                                IOOptionBits * options,
                                IOMemoryDescriptor ** memory )
{
    // Return a memory descriptor reference for some memory a client has requested 
    // be mapped into its address space.

    IOReturn ret;

    IOLog("AppleSamplePCIUserClient::clientMemoryForType(%ld)\n", type);

    switch( type ) {

        case kAppleSamplePCIMemoryType1:
            // give the client access to some shared data structure
            // (shared between this object and the client)
            fClientSharedMemory->retain();
            *memory  = fClientSharedMemory;
            ret = kIOReturnSuccess;
            break;

        case kAppleSamplePCIMemoryType2:
            // Give the client access to some of the cards memory
            // (all clients get the same)
            *memory  = fDriver->copyGlobalMemory();
            ret = kIOReturnSuccess;
            break;

        default:
            ret = kIOReturnBadArgument;
            break;
    }

    return( ret );
}

