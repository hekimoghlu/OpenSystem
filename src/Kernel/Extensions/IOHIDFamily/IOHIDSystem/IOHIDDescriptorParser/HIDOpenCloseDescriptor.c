/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
 */#include "HIDLib.h"

#if !(KERNEL || TARGET_OS_DRIVERKIT)

#include <stdlib.h>

#define IOFreeType(ptr, type)                            \
	IOFreeData(ptr, sizeof(type))

static void IOFreeData(void* ptr, size_t __unused size) {
	free(ptr);
}

#define IOMallocType(type)                              \
	IOMallocZero(sizeof(type))

static void * IOMallocZero(size_t size) {
	return calloc(1, size);
}

#else

#if TARGET_OS_DRIVERKIT
#include <DriverKit/IOLib.h>

#define IOFreeType(ptr, type)                           \
    IOSafeDeleteNULL(ptr, type, 1)

#define IOMallocType(type)                              \
    IONewZero(type, 1)

extern void IOFreeData(void * address, size_t length);

#else
#include <IOKit/system.h>
#include <IOKit/IOLib.h>
#endif

#endif

/*
 *------------------------------------------------------------------------------
 *
 * HIDCloseReportDescriptor - Close the Descriptor
 *
 *	 Input:
 *			  ptPreparsedData		- The PreParsedData Structure
 *	 Output:
 *			  ptPreparsedData		- The PreParsedData Structure
 *	 Returns:
 *			  kHIDSuccess		   - Success
 *			  kHIDNullPointerErr	  - Argument, Pointer was Null
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDCloseReportDescriptor(HIDPreparsedDataRef preparsedDataRef)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
/*
 *	Disallow NULL Pointers
*/
	if (ptPreparsedData == NULL)
		return kHIDNullPointerErr;
/*
 *	If it's marked closed then don't do anything
*/
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Free any memory that was allocated
*/
	if (ptPreparsedData->rawMemPtr != NULL)
	{
		IOFreeData(ptPreparsedData->rawMemPtr, (size_t)ptPreparsedData->numBytesAllocated);
		ptPreparsedData->rawMemPtr = NULL;
	}
/*
 *	Mark closed
*/
	ptPreparsedData->hidTypeIfValid = 0;
/*
 *	Deallocate the preparsed data
*/
	IOFreeType(ptPreparsedData, HIDPreparsedData);

	return 0;
}

/*
 *------------------------------------------------------------------------------
 *
 * HIDOpenReportDescriptor - Initialize the HID Parser
 *
 *	 Input:
 *			  psHidReportDescriptor - The HID Report Descriptor (String)
 *			  descriptorLength	   - Length of the Descriptor in bytes
 *			  ptPreparsedData		- The PreParsedData Structure
 *	 Output:
 *			  ptPreparsedData		- The PreParsedData Structure
 *	 Returns:
 *			  kHIDSuccess		   - Success
 *			  kHIDNullPointerErr	  - Argument, Pointer was Null
 *
 *------------------------------------------------------------------------------
*/
OSStatus
HIDOpenReportDescriptor	   (void *					hidReportDescriptor,
							IOByteCount 			descriptorLength,
							HIDPreparsedDataRef *	preparsedDataRef,
							UInt32					flags)
{
	HIDPreparsedDataPtr ptPreparsedData = NULL;
	OSStatus iStatus;
	HIDReportDescriptor tDescriptor;

/*
 *	Disallow NULL Pointers
*/
	if ((hidReportDescriptor == NULL) || (preparsedDataRef == NULL))
		return kHIDNullPointerErr;
	
/*
 *	Initialize the return result, and allocate space for preparsed data
*/
	*preparsedDataRef = NULL;
	
	ptPreparsedData = IOMallocType(HIDPreparsedData);
	
/*
 *	Make sure we got the memory
*/
	if (ptPreparsedData == NULL)
		return kHIDNotEnoughMemoryErr;

/*
 *	Copy the flags field
*/
	ptPreparsedData->flags = flags;
/*
 *	Initialize the memory allocation pointer
*/
	ptPreparsedData->rawMemPtr = NULL;
/*
 *	Set up the descriptor structure
*/
	tDescriptor.descriptor = hidReportDescriptor;
	tDescriptor.descriptorLength = descriptorLength;
/*
 *	Count various items in the descriptor
 *	  allocate space within the PreparsedData structure
 *	  and initialize the counters there
*/
	iStatus = HIDCountDescriptorItems(&tDescriptor,ptPreparsedData);
    
    /*
     *	Parse the Descriptor
     *	  filling in the structures in the PreparsedData structure
    */
	if (iStatus == kHIDSuccess)
    {
        iStatus = HIDParseDescriptor(&tDescriptor,ptPreparsedData);
    /*
     *	Mark the PreparsedData initialized, maybe
    */
        if (iStatus == kHIDSuccess && ptPreparsedData->rawMemPtr != NULL)
        {
            ptPreparsedData->hidTypeIfValid = kHIDOSType;
            *preparsedDataRef = (HIDPreparsedDataRef) ptPreparsedData;
        
            return kHIDSuccess;
        }
    }
    
	// something failed, deallocate everything, and make sure we return an error
    if (ptPreparsedData->rawMemPtr != NULL)
        IOFreeData(ptPreparsedData->rawMemPtr, (size_t)ptPreparsedData->numBytesAllocated);
        
    IOFreeType(ptPreparsedData, HIDPreparsedData);
    
    if (iStatus == kHIDSuccess)
        iStatus = kHIDNotEnoughMemoryErr;

	return iStatus;
}

