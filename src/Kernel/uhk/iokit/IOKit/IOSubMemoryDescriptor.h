/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#ifndef _IOSUBMEMORYDESCRIPTOR_H
#define _IOSUBMEMORYDESCRIPTOR_H

#include <IOKit/IOMemoryDescriptor.h>
#include <libkern/c++/OSPtr.h>

/*! @class IOSubMemoryDescriptor : public IOMemoryDescriptor
 *   @abstract The IOSubMemoryDescriptor object describes a memory area made up of a portion of another IOMemoryDescriptor.
 *   @discussion The IOSubMemoryDescriptor object represents a subrange of memory, specified as a portion of another IOMemoryDescriptor. */

class IOSubMemoryDescriptor : public IOMemoryDescriptor
{
	OSDeclareDefaultStructors(IOSubMemoryDescriptor);

protected:
	IOMemoryDescriptor * _parent;
	IOByteCount          _start;

	virtual void free() APPLE_KEXT_OVERRIDE;

public:
/*! @function withSubRange
 *   @abstract Create an IOMemoryDescriptor to describe a subrange of an existing descriptor.
 *   @discussion  This method creates and initializes an IOMemoryDescriptor for memory consisting of a subrange of the specified memory descriptor. The parent memory descriptor is retained by the new descriptor.
 *   @param of The parent IOMemoryDescriptor of which a subrange is to be used for the new descriptor, which will be retained by the subrange IOMemoryDescriptor.
 *   @param offset A byte offset into the parent memory descriptor's memory.
 *   @param length The length of the subrange.
 *   @param options
 *       kIOMemoryDirectionMask (options:direction)	This nibble indicates the I/O direction to be associated with the descriptor, which may affect the operation of the prepare and complete methods on some architectures.
 *   @result The created IOMemoryDescriptor on success, to be released by the caller, or zero on failure. */

	static OSPtr<IOSubMemoryDescriptor>       withSubRange(IOMemoryDescriptor *of,
	    IOByteCount offset,
	    IOByteCount length,
	    IOOptionBits options);

/*
 * Initialize or reinitialize an IOSubMemoryDescriptor to describe
 * a subrange of an existing descriptor.
 *
 * An IOSubMemoryDescriptor can be re-used by calling initSubRange
 * again on an existing instance -- note that this behavior is not
 * commonly supported in other IOKit classes, although it is here.
 */
	virtual bool initSubRange( IOMemoryDescriptor * parent,
	    IOByteCount offset, IOByteCount length,
	    IODirection withDirection );

/*
 * IOMemoryDescriptor required methods
 */

	virtual addr64_t getPhysicalSegment( IOByteCount   offset,
	    IOByteCount * length,
	    IOOptionBits  options = 0 ) APPLE_KEXT_OVERRIDE;

	virtual IOReturn prepare(IODirection forDirection = kIODirectionNone) APPLE_KEXT_OVERRIDE;

	virtual IOReturn complete(IODirection forDirection = kIODirectionNone) APPLE_KEXT_OVERRIDE;

#ifdef __LP64__
	virtual IOReturn redirect( task_t safeTask, bool redirect ) APPLE_KEXT_OVERRIDE;
#else
	IOReturn redirect( task_t safeTask, bool redirect );
#endif /* __LP64__ */

	virtual IOReturn setPurgeable( IOOptionBits newState,
	    IOOptionBits * oldState ) APPLE_KEXT_OVERRIDE;

	IOReturn setOwnership( task_t newOwner,
	    int newLedgerTag,
	    IOOptionBits newLedgerOptions );

// support map() on kIOMemoryTypeVirtual without prepare()
	virtual IOMemoryMap *       makeMapping(
		IOMemoryDescriptor *    owner,
		task_t                  intoTask,
		IOVirtualAddress        atAddress,
		IOOptionBits            options,
		IOByteCount             offset,
		IOByteCount             length ) APPLE_KEXT_OVERRIDE;

	virtual uint64_t getPreparationID( void ) APPLE_KEXT_OVERRIDE;

/*! @function getPageCounts
 *   @abstract Retrieve the number of resident and/or dirty pages encompassed by an IOMemoryDescriptor.
 *   @discussion This method returns the number of resident and/or dirty pages encompassed by an IOMemoryDescriptor.
 *   @param residentPageCount - If non-null, a pointer to a byte count that will return the number of resident pages encompassed by this IOMemoryDescriptor.
 *   @param dirtyPageCount - If non-null, a pointer to a byte count that will return the number of dirty pages encompassed by this IOMemoryDescriptor.
 *   @result An IOReturn code. */

	IOReturn getPageCounts(IOByteCount * residentPageCount,
	    IOByteCount * dirtyPageCount);
};

#endif /* !_IOSUBMEMORYDESCRIPTOR_H */
