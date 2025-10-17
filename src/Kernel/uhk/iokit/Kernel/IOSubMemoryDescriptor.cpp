/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#include <IOKit/IOSubMemoryDescriptor.h>
#include <IOKit/IOLib.h>

#include "IOKitKernelInternal.h"

#define super IOMemoryDescriptor

OSDefineMetaClassAndStructors(IOSubMemoryDescriptor, IOMemoryDescriptor)

IOReturn
IOSubMemoryDescriptor::redirect( task_t safeTask, bool doRedirect )
{
#ifdef __LP64__
	super::redirect( safeTask, doRedirect );
#endif /* __LP64__ */
	return _parent->redirect( safeTask, doRedirect );
}

IOSubMemoryDescriptor *
IOSubMemoryDescriptor::withSubRange(IOMemoryDescriptor *        of,
    IOByteCount             offset,
    IOByteCount             length,
    IOOptionBits            options)
{
	IOSubMemoryDescriptor *self = new IOSubMemoryDescriptor;

	if (self && !self->initSubRange(of, offset, length, (IODirection) options)) {
		self->release();
		self = NULL;
	}
	return self;
}

bool
IOSubMemoryDescriptor::initSubRange( IOMemoryDescriptor * parent,
    IOByteCount offset, IOByteCount length,
    IODirection direction )
{
	if (parent && ((offset + length) > parent->getLength())) {
		return false;
	}

	/*
	 * We can check the _parent instance variable before having ever set it
	 * to an initial value because I/O Kit guarantees that all our instance
	 * variables are zeroed on an object's allocation.
	 */

	if (!_parent) {
		if (!super::init()) {
			return false;
		}
	} else {
		/*
		 * An existing memory descriptor is being retargeted to
		 * point to somewhere else.  Clean up our present state.
		 */

		_parent->release();
	}

	if (parent) {
		parent->retain();
		_tag    = parent->getTag();
	} else {
		_tag    = 0;
	}
	_parent     = parent;
	_start      = offset;
	_length     = length;
	_flags      = direction;
	_flags |= kIOMemoryThreadSafe;

#ifndef __LP64__
	_direction  = (IODirection) (_flags & kIOMemoryDirectionMask);
#endif /* !__LP64__ */

	return true;
}

void
IOSubMemoryDescriptor::free( void )
{
	if (_parent) {
		_parent->release();
	}

	super::free();
}

addr64_t
IOSubMemoryDescriptor::getPhysicalSegment(IOByteCount offset, IOByteCount * length, IOOptionBits options)
{
	addr64_t    address;
	IOByteCount actualLength;

	assert(offset <= _length);

	if (length) {
		*length = 0;
	}

	if (offset >= _length) {
		return 0;
	}

	address = _parent->getPhysicalSegment( offset + _start, &actualLength, options );

	if (address && length) {
		*length = min( _length - offset, actualLength );
	}

	return address;
}

IOReturn
IOSubMemoryDescriptor::setPurgeable( IOOptionBits newState,
    IOOptionBits * oldState )
{
	IOReturn err;

	err = _parent->setPurgeable( newState, oldState );

	return err;
}

IOReturn
IOSubMemoryDescriptor::setOwnership( task_t newOwner,
    int newLedgerTag,
    IOOptionBits newLedgerOptions )
{
	IOReturn err;

	if (iokit_iomd_setownership_enabled == FALSE) {
		return kIOReturnUnsupported;
	}

	err = _parent->setOwnership( newOwner, newLedgerTag, newLedgerOptions );

	return err;
}

IOReturn
IOSubMemoryDescriptor::prepare(
	IODirection forDirection)
{
	IOReturn    err;

	err = _parent->prepare( forDirection);

	return err;
}

IOReturn
IOSubMemoryDescriptor::complete(
	IODirection forDirection)
{
	IOReturn    err;

	err = _parent->complete( forDirection);

	return err;
}

IOMemoryMap *
IOSubMemoryDescriptor::makeMapping(
	IOMemoryDescriptor *    owner,
	task_t                  intoTask,
	IOVirtualAddress        address,
	IOOptionBits            options,
	IOByteCount             offset,
	IOByteCount             length )
{
	IOMemoryMap * mapping = NULL;

#ifndef __LP64__
	if (!(kIOMap64Bit & options)) {
		panic("IOSubMemoryDescriptor::makeMapping !64bit");
	}
#endif /* !__LP64__ */

	mapping = (IOMemoryMap *) _parent->makeMapping(
		owner,
		intoTask,
		address,
		options, _start + offset, length );

	return mapping;
}

uint64_t
IOSubMemoryDescriptor::getPreparationID( void )
{
	uint64_t pID;

	if (!super::getKernelReserved()) {
		return kIOPreparationIDUnsupported;
	}

	pID = _parent->getPreparationID();
	if (reserved->kernReserved[0] != pID) {
		reserved->kernReserved[0] = pID;
		reserved->preparationID   = kIOPreparationIDUnprepared;
		super::setPreparationID();
	}

	return super::getPreparationID();
}

IOReturn
IOSubMemoryDescriptor::getPageCounts(IOByteCount * residentPageCount,
    IOByteCount * dirtyPageCount)
{
	return _parent->getPageCounts(residentPageCount, dirtyPageCount);
}
