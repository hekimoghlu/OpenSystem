/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
 *  IOFWBufferQ.h
 *  IOFireWireFamily
 *
 *  Created by calderon on 9/17/09.
 *  Copyright 2009 Apple. All rights reserved.
 *
 */

#ifndef __IOFWRingBufferQ_H__
#define __IOFWRingBufferQ_H__

// public
#import <IOKit/IOMemoryDescriptor.h>

//using namespace IOFireWireLib;

// IOFWRingBufferQ
// Description: A ring buffered FIFO queue
	
class IOFWRingBufferQ: public OSObject
{
	OSDeclareDefaultStructors(IOFWRingBufferQ);
	
public:
	
	static IOFWRingBufferQ *	withAddressRange( mach_vm_address_t address, mach_vm_size_t length, IOOptionBits options, task_t task );
	
	virtual bool			initQ( mach_vm_address_t address, mach_vm_size_t length, IOOptionBits options, task_t task );
	virtual void			free( void ) APPLE_KEXT_OVERRIDE;
	virtual bool			isEmpty( void );
	virtual bool			dequeueBytes( IOByteCount size );
	virtual bool			dequeueBytesWithCopy( void * copy, IOByteCount size );
	virtual IOByteCount		readBytes(IOByteCount offset, void * bytes, IOByteCount withLength);
	virtual bool			enqueueBytes( void * bytes, IOByteCount size );
	virtual bool			isSpaceAvailable( IOByteCount size, IOByteCount * offset );
	virtual bool			front( void * copy, IOByteCount size, IOByteCount * paddingBytes );
	virtual IOByteCount		spaceAvailable( void );
	virtual bool			willFitAtEnd( IOByteCount sizeOfEntry, IOByteCount * offset, IOByteCount * paddingBytes );
	virtual IOByteCount		frontEntryOffset( IOByteCount sizeOfEntry, IOByteCount * paddingBytes );
	
private:
	IOMemoryDescriptor *			fMemDescriptor;
	bool							fMemDescriptorPrepared;
	IOByteCount						fBufferSize;
	IOByteCount						fFrontOffset;
	IOByteCount						fQueueLength;
} ;

#endif //__IOFWRingBufferQ_H__
