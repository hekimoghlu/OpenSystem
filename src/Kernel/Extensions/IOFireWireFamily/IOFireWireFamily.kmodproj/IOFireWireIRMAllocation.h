/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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
#ifndef _IOKIT_IOFIREWIREIRMALLOCATION_H
#define _IOKIT_IOFIREWIREIRMALLOCATION_H

#include <libkern/c++/OSObject.h>

class IOFireWireController;

//#include <IOKit/firewire/IOFireWireController.h>
//#include <IOKit/firewire/IOFireWireFamilyCommon.h>

/*! @class IOFireWireIRMAllocation
*/

class IOFireWireIRMAllocation : public OSObject
{
	friend class IOFireWireController;

    OSDeclareDefaultStructors(IOFireWireIRMAllocation)
	
public:
		
		// Prototype for the callback if reallocation after bus-reset is unsuccessful.
		typedef IOReturn (*AllocationLostNotificationProc)(void* refCon, class IOFireWireIRMAllocation* allocation);

	// Initialize the IRM allocation object. 
    virtual bool init( IOFireWireController * control,
					   Boolean releaseIRMResourcesOnFree = true, 
					   AllocationLostNotificationProc allocationLostProc = NULL,
					   void *pLostProcRefCon = NULL);
    	
	// Specify whether of not the IRM resources should automatically
	// be released when freeing this allocation object.
	virtual void setReleaseIRMResourcesOnFree(Boolean doRelease);
	
	// Use the IRMAllocation object to allocate isoch resources
	virtual IOReturn allocateIsochResources(UInt8 isochChannel, UInt32 bandwidthUnits);

	// Free isoch resources previously allocated with a call to allocateIsochResources
	virtual IOReturn deallocateIsochResources(void);
	
	// Returns true if isoch resources are currently allocated, and if true, the allocated channel, and amount of isoch bandwidth.
	virtual Boolean areIsochResourcesAllocated(UInt8 *pAllocatedIsochChannel, UInt32 *pAllocatedBandwidthUnits);
	
	// Get the refcon
	virtual void * GetRefCon(void);
	virtual void SetRefCon(void* refCon); 

	// Override the base-class release function for special processing
	virtual void release() const APPLE_KEXT_OVERRIDE;
	
protected:

		/*! @struct ExpansionData
		@discussion This structure will be used to expand the capablilties of the class in the future.
		*/    
		struct ExpansionData { };
	
		/*! @var reserved
		Reserved for future use.  (Internal use only)  */
		ExpansionData *reserved;

		// Free the allocation object (and release IRM resources if needed)
		virtual void free( void ) APPLE_KEXT_OVERRIDE;

		// Controller will call this to notify about bus-reset complete.
		virtual void handleBusReset(UInt32 generation);
	
		virtual void failedToRealloc(void);
		virtual UInt32 getAllocationGeneration(void);
		static void threadFunc( void * arg );

private:
	
	AllocationLostNotificationProc fAllocationLostProc;
	void *fLostProcRefCon;
	Boolean fReleaseIRMResourcesOnFree;
	UInt8 fIsochChannel; 
	UInt32 fBandwidthUnits;
	UInt32 fAllocationGeneration;
	IORecursiveLock *fLock ;
	IOFireWireController *fControl;
	Boolean isAllocated;
	
	OSMetaClassDeclareReservedUnused(IOFireWireIRMAllocation, 0);
	OSMetaClassDeclareReservedUnused(IOFireWireIRMAllocation, 1);
	OSMetaClassDeclareReservedUnused(IOFireWireIRMAllocation, 2);
	OSMetaClassDeclareReservedUnused(IOFireWireIRMAllocation, 3);
	OSMetaClassDeclareReservedUnused(IOFireWireIRMAllocation, 4);
	OSMetaClassDeclareReservedUnused(IOFireWireIRMAllocation, 5);
	OSMetaClassDeclareReservedUnused(IOFireWireIRMAllocation, 6);
	OSMetaClassDeclareReservedUnused(IOFireWireIRMAllocation, 7);
};

#endif // _IOKIT_IOFIREWIREIRMALLOCATION_H
