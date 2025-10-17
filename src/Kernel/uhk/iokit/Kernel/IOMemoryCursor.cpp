/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
/* IOMemoryCursor.cpp created by wgulland on 1999-3-02 */

#define IOKIT_ENABLE_SHARED_PTR

#include <IOKit/assert.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOMemoryCursor.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <libkern/OSByteOrder.h>

/**************************** class IOMemoryCursor ***************************/

#undef super
#define super OSObject
OSDefineMetaClassAndStructors(IOMemoryCursor, OSObject)

OSSharedPtr<IOMemoryCursor>
IOMemoryCursor::withSpecification(SegmentFunction  inSegFunc,
    IOPhysicalLength inMaxSegmentSize,
    IOPhysicalLength inMaxTransferSize,
    IOPhysicalLength inAlignment)
{
	OSSharedPtr<IOMemoryCursor> me = OSMakeShared<IOMemoryCursor>();

	if (me && !me->initWithSpecification(inSegFunc,
	    inMaxSegmentSize,
	    inMaxTransferSize,
	    inAlignment)) {
		return nullptr;
	}

	return me;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

bool
IOMemoryCursor::initWithSpecification(SegmentFunction  inSegFunc,
    IOPhysicalLength inMaxSegmentSize,
    IOPhysicalLength inMaxTransferSize,
    IOPhysicalLength inAlignment)
{
// @@@ gvdl: Remove me
#if 1
	static UInt sMaxDBDMASegment;
	if (!sMaxDBDMASegment) {
		sMaxDBDMASegment = (UInt) - 1;
		if (PE_parse_boot_argn("mseg", &sMaxDBDMASegment, sizeof(sMaxDBDMASegment))) {
			IOLog("Setting MaxDBDMASegment to %d\n", sMaxDBDMASegment);
		}
	}

	if (inMaxSegmentSize > sMaxDBDMASegment) {
		inMaxSegmentSize = sMaxDBDMASegment;
	}
#endif

	if (!super::init()) {
		return false;
	}

	if (!inSegFunc) {
		return false;
	}
	if (inMaxTransferSize > UINT_MAX) {
		return false;
	}

	outSeg              = inSegFunc;
	maxSegmentSize      = inMaxSegmentSize;
	if (inMaxTransferSize) {
		maxTransferSize = inMaxTransferSize;
	} else {
		maxTransferSize = (IOPhysicalLength) - 1;
	}
	alignMask           = inAlignment - 1;
	assert(alignMask == 0);         // No alignment code yet!

	return true;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

UInt32
IOMemoryCursor::genPhysicalSegments(IOMemoryDescriptor *inDescriptor,
    IOByteCount         fromPosition,
    void *              inSegments,
    UInt32              inMaxSegments,
    UInt32              inMaxTransferSize,
    IOByteCount         *outTransferSize)
{
	if (!inDescriptor) {
		return 0;
	}

	if (!inMaxSegments) {
		return 0;
	}

	if (!inMaxTransferSize) {
		inMaxTransferSize = (typeof(inMaxTransferSize))maxTransferSize;
	}

	/*
	 * Iterate over the packet, translating segments where allowed
	 *
	 * If we finished cleanly return number of segments found
	 * and update the position in the descriptor.
	 */
	PhysicalSegment curSeg = { 0, 0 };
	UInt curSegIndex = 0;
	UInt curTransferSize = 0;
	IOByteCount inDescriptorLength = inDescriptor->getLength();
	PhysicalSegment seg = { 0, 0 };

	while ((seg.location) || (fromPosition < inDescriptorLength)) {
		if (!seg.location) {
			seg.location = inDescriptor->getPhysicalSegment(
				fromPosition, (IOByteCount*)&seg.length);
			assert(seg.location);
			assert(seg.length);
			fromPosition += seg.length;
		}

		if (!curSeg.location) {
			curTransferSize += seg.length;
			curSeg = seg;
			seg.location = 0;
		} else if ((curSeg.location + curSeg.length == seg.location)) {
			curTransferSize += seg.length;
			curSeg.length += seg.length;
			seg.location = 0;
		}

		if (!seg.location) {
			if ((curSeg.length > maxSegmentSize)) {
				seg.location = curSeg.location + maxSegmentSize;
				seg.length = curSeg.length - maxSegmentSize;
				curTransferSize -= seg.length;
				curSeg.length -= seg.length;
			}

			if ((curTransferSize >= inMaxTransferSize)) {
				curSeg.length -= curTransferSize - inMaxTransferSize;
				curTransferSize = inMaxTransferSize;
				break;
			}
		}

		if (seg.location) {
			if ((curSegIndex + 1 == inMaxSegments)) {
				break;
			}
			(*outSeg)(curSeg, inSegments, curSegIndex++);
			curSeg.location = 0;
		}
	}

	if (curSeg.location) {
		(*outSeg)(curSeg, inSegments, curSegIndex++);
	}

	if (outTransferSize) {
		*outTransferSize = curTransferSize;
	}

	return curSegIndex;
}

/************************ class IONaturalMemoryCursor ************************/

#undef super
#define super IOMemoryCursor
OSDefineMetaClassAndStructors(IONaturalMemoryCursor, IOMemoryCursor)

void
IONaturalMemoryCursor::outputSegment(PhysicalSegment segment,
    void *          outSegments,
    UInt32          outSegmentIndex)
{
	((PhysicalSegment *) outSegments)[outSegmentIndex] = segment;
}

OSSharedPtr<IONaturalMemoryCursor>
IONaturalMemoryCursor::withSpecification(IOPhysicalLength inMaxSegmentSize,
    IOPhysicalLength inMaxTransferSize,
    IOPhysicalLength inAlignment)
{
	OSSharedPtr<IONaturalMemoryCursor> me = OSMakeShared<IONaturalMemoryCursor>();

	if (me && !me->initWithSpecification(inMaxSegmentSize,
	    inMaxTransferSize,
	    inAlignment)) {
		return nullptr;
	}

	return me;
}

bool
IONaturalMemoryCursor::initWithSpecification(IOPhysicalLength inMaxSegmentSize,
    IOPhysicalLength inMaxTransferSize,
    IOPhysicalLength inAlignment)
{
	return super::initWithSpecification(&IONaturalMemoryCursor::outputSegment,
	           inMaxSegmentSize,
	           inMaxTransferSize,
	           inAlignment);
}

/************************** class IOBigMemoryCursor **************************/

#undef super
#define super IOMemoryCursor
OSDefineMetaClassAndStructors(IOBigMemoryCursor, IOMemoryCursor)

void
IOBigMemoryCursor::outputSegment(PhysicalSegment inSegment,
    void *          inSegments,
    UInt32          inSegmentIndex)
{
	IOPhysicalAddress * segment;

	segment = &((PhysicalSegment *) inSegments)[inSegmentIndex].location;
#if IOPhysSize == 64
	OSWriteBigInt64(segment, 0, inSegment.location);
	OSWriteBigInt64(segment, sizeof(IOPhysicalAddress), inSegment.length);
#else
	OSWriteBigInt(segment, 0, inSegment.location);
	OSWriteBigInt(segment, sizeof(IOPhysicalAddress), inSegment.length);
#endif
}

OSSharedPtr<IOBigMemoryCursor>
IOBigMemoryCursor::withSpecification(IOPhysicalLength inMaxSegmentSize,
    IOPhysicalLength inMaxTransferSize,
    IOPhysicalLength inAlignment)
{
	OSSharedPtr<IOBigMemoryCursor> me = OSMakeShared<IOBigMemoryCursor>();

	if (me && !me->initWithSpecification(inMaxSegmentSize,
	    inMaxTransferSize,
	    inAlignment)) {
		return nullptr;
	}

	return me;
}

bool
IOBigMemoryCursor::initWithSpecification(IOPhysicalLength inMaxSegmentSize,
    IOPhysicalLength inMaxTransferSize,
    IOPhysicalLength inAlignment)
{
	return super::initWithSpecification(&IOBigMemoryCursor::outputSegment,
	           inMaxSegmentSize,
	           inMaxTransferSize,
	           inAlignment);
}

/************************* class IOLittleMemoryCursor ************************/

#undef super
#define super IOMemoryCursor
OSDefineMetaClassAndStructors(IOLittleMemoryCursor, IOMemoryCursor)

void
IOLittleMemoryCursor::outputSegment(PhysicalSegment inSegment,
    void *          inSegments,
    UInt32          inSegmentIndex)
{
	IOPhysicalAddress * segment;

	segment = &((PhysicalSegment *) inSegments)[inSegmentIndex].location;
#if IOPhysSize == 64
	OSWriteLittleInt64(segment, 0, inSegment.location);
	OSWriteLittleInt64(segment, sizeof(IOPhysicalAddress), inSegment.length);
#else
	OSWriteLittleInt(segment, 0, inSegment.location);
	OSWriteLittleInt(segment, sizeof(IOPhysicalAddress), inSegment.length);
#endif
}

OSSharedPtr<IOLittleMemoryCursor>
IOLittleMemoryCursor::withSpecification(IOPhysicalLength inMaxSegmentSize,
    IOPhysicalLength inMaxTransferSize,
    IOPhysicalLength inAlignment)
{
	OSSharedPtr<IOLittleMemoryCursor> me = OSMakeShared<IOLittleMemoryCursor>();

	if (me && !me->initWithSpecification(inMaxSegmentSize,
	    inMaxTransferSize,
	    inAlignment)) {
		return nullptr;
	}

	return me;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

bool
IOLittleMemoryCursor::initWithSpecification(IOPhysicalLength inMaxSegmentSize,
    IOPhysicalLength inMaxTransferSize,
    IOPhysicalLength inAlignment)
{
	return super::initWithSpecification(&IOLittleMemoryCursor::outputSegment,
	           inMaxSegmentSize,
	           inMaxTransferSize,
	           inAlignment);
}
