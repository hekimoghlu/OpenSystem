/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
 * Copyright (c) 1998 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */

#ifndef _IOKIT_IODEVICEMEMORY_H
#define _IOKIT_IODEVICEMEMORY_H

#include <IOKit/IOMemoryDescriptor.h>
#include <libkern/c++/OSPtr.h>

/*! @class IODeviceMemory
 *   @abstract An IOMemoryDescriptor used for device physical memory ranges.
 *   @discussion The IODeviceMemory class is a simple subclass of IOMemoryDescriptor that uses its methods to describe a single range of physical memory on a device. IODeviceMemory objects are usually looked up with IOService or IOPCIDevice accessors, and are created by memory-mapped bus families. IODeviceMemory implements only some factory methods in addition to the methods of IOMemoryDescriptor.
 */

class IODeviceMemory : public IOMemoryDescriptor
{
	OSDeclareDefaultStructors(IODeviceMemory);

public:

/*! @struct InitElement
 *   @field start First physical address in the range.
 *   @field length Length of the range.
 *   @field tag 32-bit value not interpreted by IODeviceMemory or IOMemoryDescriptor, for use by the bus family. */

	struct InitElement {
		IOPhysicalAddress       start;
		IOPhysicalLength        length;
		IOOptionBits            tag;
	};

/*! @function arrayFromList
 *   @abstract Constructs an OSArray of IODeviceMemory instances, each describing one physical range, and a tag value.
 *   @discussion This method creates IODeviceMemory instances for each physical range passed in an IODeviceMemory::InitElement array. Each element consists of a physical address, length and tag value for the IODeviceMemory. The instances are returned as a created OSArray.
 *   @param list An array of IODeviceMemory::InitElement structures.
 *   @param count The number of elements in the list.
 *   @result Returns a created OSArray of IODeviceMemory objects, to be released by the caller, or zero on failure. */

	static OSPtr<OSArray>             arrayFromList(
		InitElement             list[],
		IOItemCount             count );

/*! @function withRange
 *   @abstract Constructs an IODeviceMemory instance, describing one physical range.
 *   @discussion This method creates an IODeviceMemory instance for one physical range passed as a physical address and length. It just calls IOMemoryDescriptor::withPhysicalAddress.
 *   @param start The physical address of the first byte in the memory.
 *   @param length The length of memory.
 *   @result Returns the created IODeviceMemory on success, to be released by the caller, or zero on failure. */

	static OSPtr<IODeviceMemory>      withRange(
		IOPhysicalAddress       start,
		IOPhysicalLength        length );

/*! @function withSubRange
 *   @abstract Constructs an IODeviceMemory instance, describing a subset of an existing IODeviceMemory range.
 *   @discussion This method creates an IODeviceMemory instance for a subset of an existing IODeviceMemory range, passed as a physical address offset and length. It just calls IOMemoryDescriptor::withSubRange.
 *   @param of The parent IODeviceMemory of which a subrange is to be used for the new descriptor, which will be retained by the subrange IODeviceMemory.
 *   @param offset A byte offset into the parent's memory.
 *   @param length The length of the subrange.
 *   @result Returns the created IODeviceMemory on success, to be released by the caller, or zero on failure. */

	static OSPtr<IODeviceMemory>      withSubRange(
		IODeviceMemory *        of,
		IOPhysicalAddress       offset,
		IOPhysicalLength        length );
};

#endif /* ! _IOKIT_IODEVICEMEMORY_H */
