/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
#include <os/log.h>

#include <DriverKit/IOLib.h>
#include <DriverKit/IOMemoryMap.h>
#include <BlockStorageDeviceDriverKit/IOUserBlockStorageDevice.h>
#include <DriverKit/IODispatchQueue.h>

#undef super
#define super IOService

struct IOUserBlockStorageDevice_IVars
{
	IODispatchQueue *fCompletionQueue;
};

bool IOUserBlockStorageDevice::init()
{
	if (!super::init())
		return false;

	ivars = IONewZero(IOUserBlockStorageDevice_IVars, 1);
	if (!ivars)
		return false;

	return true;

}

void IOUserBlockStorageDevice::free()
{
	IODelete ( ivars, IOUserBlockStorageDevice_IVars, 1 );
	super::free ( );
}

kern_return_t
IMPL(IOUserBlockStorageDevice, DoAsyncUnmap)
{
	IOMemoryMap *map;
	kern_return_t retVal;

	retVal = buffer->CreateMapping(0,0,0,0,0,&map);
	if (retVal != kIOReturnSuccess) {
		os_log(OS_LOG_DEFAULT, "IOUserBlockStorageDevice::DoAsyncUnmap() - Failed to map the buffer");
		goto out;
	}

	retVal = DoAsyncUnmapPriv(requestID, (struct BlockRange *)map->GetAddress(), numOfRanges);

	OSSafeReleaseNULL(map);
out:
	return retVal;
}

kern_return_t IMPL(IOUserBlockStorageDevice, Stop)
{
	os_log(OS_LOG_DEFAULT, "IOUserBlockStorageDevice::Stop()");
    
    void  ( ^finalize )( void ) = ^{
 
        ivars->fCompletionQueue->release();
        
        Stop ( provider, SUPERDISPATCH );
        
    };

    /* Stop dispatch queue */
    ivars->fCompletionQueue->Cancel ( finalize );
 
    return kIOReturnSuccess;
}

kern_return_t IMPL(IOUserBlockStorageDevice, Start)
{
	kern_return_t retVal;

	os_log(OS_LOG_DEFAULT, "IOUserBlockStorageDevice::Start()");

	retVal = IODispatchQueue::Create ( "CompletionQueue", 0, 0, &ivars->fCompletionQueue );
	if (retVal != kIOReturnSuccess) {
		os_log(OS_LOG_DEFAULT, "IOUserBlockStorageDevice::Start() - failed to create dispatch queue");
		goto DispatchQueueCreateError;
	}

	retVal = SetDispatchQueue ( "Completion", ivars->fCompletionQueue );
	if (retVal != kIOReturnSuccess) {
		os_log(OS_LOG_DEFAULT, "IOUserBlockStorageDevice::Start() - failed to set dispatch queue");
        goto Error;
	}

    struct DeviceParams deviceParams;
    retVal = GetDeviceParams(&deviceParams);
    if (retVal != kIOReturnSuccess){
        os_log(OS_LOG_DEFAULT, "IOUserBlockStorageDevice::Start() - failed to get device params");
        goto Error;
    }

	retVal = StartDev(provider, &deviceParams);
	if (retVal != kIOReturnSuccess){
		os_log(OS_LOG_DEFAULT, "IOUserBlockStorageDevice::Start() - failed to start device in kernel");
        goto Error;
	}
	retVal = RegisterDext();

	if (retVal != kIOReturnSuccess) {
		os_log(OS_LOG_DEFAULT, "IOUserBlockStorageDevice::Start() - failed to set register dext");
        goto Error;
	}

	return Start(provider, SUPERDISPATCH);

Error:
    ivars->fCompletionQueue->release();
DispatchQueueCreateError:
    return retVal;
}
