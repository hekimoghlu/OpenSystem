/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
#ifndef _IOUSERBLOCKSTORAGEDEVICE_KEXT_H
#define _IOUSERBLOCKSTORAGEDEVICE_KEXT_H

#include <IOKit/storage/IOBlockStorageDevice.h>
#include <IOKit/IOLocks.h>
#include "IOUserBlockStorageDevice.h"
#include "IORequestsPool.h"
#include <stdatomic.h>

class IOPerfControlClient;
class IORequest;
class IOMapper;
class IOUserBlockStorageDevice : public IOBlockStorageDevice
{
	OSDeclareDefaultStructorsWithDispatch ( IOUserBlockStorageDevice );
public:
	virtual bool init ( OSDictionary * dictionary = NULL ) APPLE_KEXT_OVERRIDE;
	virtual void free( void ) APPLE_KEXT_OVERRIDE;
	virtual bool start (IOService* provider) APPLE_KEXT_OVERRIDE;
	virtual void stop (IOService* provider) APPLE_KEXT_OVERRIDE;
	virtual bool willTerminate( IOService * provider, IOOptionBits options ) APPLE_KEXT_OVERRIDE;

	IOReturn doAsyncReadWrite (IOMemoryDescriptor *inBuffer,
				   uint64_t inLBA,
				   uint64_t inBlockCount,
				   IOStorageAttributes *inAttributes,
				   IOStorageCompletion *inCompletion ) APPLE_KEXT_OVERRIDE;

	IOReturn doSynchronize(UInt64 block, UInt64 nblks, IOStorageSynchronizeOptions options) APPLE_KEXT_OVERRIDE;
	IOReturn doUnmap(IOBlockStorageDeviceExtent *extents,
			 UInt32 extentsCount,
			 IOStorageUnmapOptions options = 0) APPLE_KEXT_OVERRIDE;
	IOReturn doEjectMedia ( void ) APPLE_KEXT_OVERRIDE;

	virtual IOReturn doFormatMedia ( uint64_t inByteCapacity ) APPLE_KEXT_OVERRIDE;
	virtual UInt32 doGetFormatCapacities ( UInt64 *ioCapacities,
					 UInt32  inCapacitiesMaxCount ) const
	    APPLE_KEXT_OVERRIDE;

	virtual char *getVendorString ( void ) APPLE_KEXT_OVERRIDE;
	virtual char *getProductString ( void ) APPLE_KEXT_OVERRIDE;
	virtual char *getRevisionString ( void ) APPLE_KEXT_OVERRIDE;
	virtual char *getAdditionalDeviceInfoString ( void ) APPLE_KEXT_OVERRIDE;

	virtual IOReturn reportMaxValidBlock ( uint64_t *ioMaxBlock ) APPLE_KEXT_OVERRIDE;
	virtual IOReturn reportRemovability ( bool *ioIsRemovable ) APPLE_KEXT_OVERRIDE;
	virtual IOReturn reportWriteProtection ( bool *ioIsWriteProtected ) APPLE_KEXT_OVERRIDE;

	virtual IOReturn reportBlockSize ( uint64_t *ioBlockSize ) APPLE_KEXT_OVERRIDE;
	virtual IOReturn reportEjectability ( bool *ioIsEjectable ) APPLE_KEXT_OVERRIDE;
	virtual IOReturn reportMediaState ( bool *ioMediaPresent,
				    bool *ioChanged ) APPLE_KEXT_OVERRIDE;
    IOPerfControlClient *getPerfControlClient();
private:
	IORequest *removeOutstandingRequestAndMarkSlotFree(uint32_t index);
	IORequest *removeOutstandingRequestAndMarkSlotAborted(uint32_t index);
	bool addOutstandingRequestAndMarkSlotOccupied(IORequest * request);
	bool markSlotFree(IORequest *request);

private:
	IOMapper *fMapper;
	IORequestsPool fRequestsPool;

	/* Array of outstanding requests */
	IORequest * _Atomic *fOutstandingRequests;

	struct DeviceParams fDeviceParams;
    bool fIsEjectable;

	OSString *fVendorName;
	OSString *fProductName;
	OSString *fRevision;
	OSString *fAdditionDeviceInfo;

	IOPerfControlClient * fPerfControlClient;
	bool fPoolInitialized;

    /*
     fSkipStartDev: This flag's purpose is to help us distingues which path are we in and whether should we skip startDev.
     There are two flows that could be  - Software driver and an actual HW device (NVMe driver for instance).

     Software driver flow:
        The user is calling Start_Impl (user space) which in it's turn will call startDev (kernel space) and do its magic. in this case we shouldn't ignore StartDev. StartDev is allocating and initializing IOUserBLockStorageDevice members based on deviceParams which in this case is populated. Thus, the flag will stay false and we shouldn't ignore StartDev.

     HW device:
        When attaching an HW device, it triggers the init->attach->start regular procedure. However, not like in the SW case, the start method (kernel space) is calling to the derived class Start_Impl method (user space - using rpc) which only at the end populates the relevant properties for us to use in StartDev for further initialization.
            Due to this mechanism, we have to firstly ignore StartDev, and only after the derived class Start_Impl method ends, we will call StartDev to do its magic.

     Because of those different flows and use cases of IOUserBlockStorageDevice class, we are using a flag for indication of which path are we in currently, and act accordingly to that.
     */
    bool fSkipStartDev;
    bool fStartDevStarted;
    bool fHWPath;
};

#endif /* _IOUSERBLOCKSTORAGEDEVICE_KEXT_H */
