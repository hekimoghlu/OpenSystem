/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
#define IOKIT_ENABLE_SHARED_PTR
#include <IOKit/IOGuardPageMemoryDescriptor.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IOLib.h>
#include <vm/vm_kern_xnu.h>
#include <mach/mach_vm.h>

#define super IOGeneralMemoryDescriptor

OSDefineMetaClassAndStructorsWithZone(IOGuardPageMemoryDescriptor, IOGeneralMemoryDescriptor, ZC_ZFREE_CLEARMEM);

OSSharedPtr<IOGuardPageMemoryDescriptor>
IOGuardPageMemoryDescriptor::withSize(vm_size_t size)
{
	OSSharedPtr<IOGuardPageMemoryDescriptor> me = OSMakeShared<IOGuardPageMemoryDescriptor>();

	if (me && !me->initWithSize(size)) {
		me.reset();
	}
	return me;
}

bool
IOGuardPageMemoryDescriptor::initWithSize(vm_size_t size)
{
	mach_vm_address_t address;
	kern_return_t kr;
	IOOptionBits  iomdOptions = kIOMemoryTypeVirtual64 | kIOMemoryAsReference | kIODirectionOutIn;

	size = round_page(size);

	_ranges.v64 = IOMallocType(IOAddressRange);
	if (!_ranges.v64) {
		return false;
	}

	kr = mach_vm_allocate_kernel(kernel_map, &address, size,
	    VM_MAP_KERNEL_FLAGS_ANYWHERE(.vm_tag = VM_KERN_MEMORY_IOKIT));
	if (kr != KERN_SUCCESS) {
		return false;
	}


	_ranges.v64->address = address;
	_ranges.v64->length  = size;

	if (!super::initWithOptions(_ranges.v64, 1, 0, kernel_task, iomdOptions, NULL)) {
		return false;
	}

	_size = size;
	_buffer = (vm_offset_t)address;

	return true;
}

void
IOGuardPageMemoryDescriptor::free()
{
	if (_buffer) {
		vm_deallocate(kernel_map, _buffer, _size);
		_buffer = 0;
	}

	if (_ranges.v64) {
		IOFreeType(_ranges.v64, IOAddressRange);
	}

	super::free();
}

IOReturn
IOGuardPageMemoryDescriptor::doMap(vm_map_t           addressMap,
    IOVirtualAddress * atAddress,
    IOOptionBits       options,
    IOByteCount        sourceOffset,
    IOByteCount        length)
{
	IOReturn ret = super::doMap(addressMap, atAddress, options, sourceOffset, length);
	if (ret == kIOReturnSuccess) {
		IOMemoryMap *     mapping = (IOMemoryMap *) *atAddress;
		vm_map_t          map     = mapping->fAddressMap;
		mach_vm_size_t    length  = mapping->fLength;
		mach_vm_address_t address = mapping->fAddress;
		kern_return_t kr = mach_vm_protect(map, address, length, true, VM_PROT_NONE);
		if (kr != KERN_SUCCESS) {
			doUnmap(map, (IOVirtualAddress) mapping, 0);
			return kIOReturnError;
		}
	}
	return ret;
}
