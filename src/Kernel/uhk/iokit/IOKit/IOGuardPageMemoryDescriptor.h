/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#ifndef _IOGUARDPAGEMEMORYDESCRIPTOR_H
#define _IOGUARDPAGEMEMORYDESCRIPTOR_H

#include <IOKit/IOMemoryDescriptor.h>
#include <libkern/c++/OSPtr.h>

/*!
 *   @class IOGuardPageMemoryDescriptor
 *   @abstract Provides a memory descriptor that allows for variable size guard regions. Use with
 *             IOMultiMemoryDescriptor to surround other memory descriptors with guard pages.
 */
class IOGuardPageMemoryDescriptor : public IOGeneralMemoryDescriptor
{
	OSDeclareDefaultStructors(IOGuardPageMemoryDescriptor);

protected:
	virtual void free() APPLE_KEXT_OVERRIDE;

	vm_offset_t    _buffer;
	vm_size_t      _size;

public:

	/* @function withSize
	 *  @discussion Create a IOGuardPageMemoryDescriptor with the specified size.
	 *  @param Size of the guard region. This will be rounded up to the nearest multiple of page size.
	 *  @return IOGuardPageMemoryDescriptor instance to be released by the caller, which will free the allocated
	 *          virtual memory region.
	 */
	static OSPtr<IOGuardPageMemoryDescriptor> withSize(vm_size_t size);

	virtual bool initWithSize(vm_size_t size);

private:
	virtual IOReturn doMap(vm_map_t           addressMap,
	    IOVirtualAddress * atAddress,
	    IOOptionBits       options,
	    IOByteCount        sourceOffset = 0,
	    IOByteCount        length = 0 ) APPLE_KEXT_OVERRIDE;
};

#endif /* !_IOGUARDPAGEMEMORYDESCRIPTOR_H */
