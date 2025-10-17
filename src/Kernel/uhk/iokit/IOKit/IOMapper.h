/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
#ifndef __IOKIT_IOMAPPER_H
#define __IOKIT_IOMAPPER_H

#include <sys/cdefs.h>
#include <IOKit/IOTypes.h>
#include <mach/vm_types.h>

__BEGIN_DECLS

// These are C accessors to the system mapper for non-IOKit clients
ppnum_t IOMapperIOVMAlloc(unsigned pages);
void IOMapperIOVMFree(ppnum_t addr, unsigned pages);
ppnum_t IOMapperInsertPage(ppnum_t addr, unsigned offset, ppnum_t page);

__END_DECLS

#if __cplusplus

#include <IOKit/IOService.h>
#include <IOKit/IOMemoryDescriptor.h>
#include <IOKit/IODMACommand.h>
#include <libkern/c++/OSPtr.h>

class OSData;

extern const OSSymbol * gIOMapperIDKey;

class IOMapper : public IOService
{
	OSDeclareAbstractStructors(IOMapper);

// Give the platform expert access to setMapperRequired();
	friend class IOPlatformExpert;
	friend class IOMemoryDescriptor;
	friend class IOGeneralMemoryDescriptor;

private:
	enum SystemMapperState {
		kNoMapper  = 0,
		kUnknown   = 1,
		kHasMapper = 2, // Any other value is pointer to a live mapper
		kWaitMask  = 3,
	};
protected:
#ifdef XNU_KERNEL_PRIVATE
	uint64_t   __reservedA[6];
	kern_allocation_name_t fAllocName;
	uint32_t   __reservedB;
	uint32_t   fPageSize;
#else
	uint64_t __reserved[8];
#endif
	bool fIsSystem;

	static void setMapperRequired(bool hasMapper);
	static void waitForSystemMapper();

	virtual bool initHardware(IOService *provider) = 0;

public:
	virtual bool start(IOService *provider) APPLE_KEXT_OVERRIDE;
	virtual void free() APPLE_KEXT_OVERRIDE;

// To get access to the system mapper IOMapper::gSystem
	static IOMapper *gSystem;

	static void
	checkForSystemMapper()
	{
		if ((uintptr_t) gSystem & kWaitMask) {
			waitForSystemMapper();
		}
	}

	static OSPtr<IOMapper>  copyMapperForDevice(IOService * device);
	static OSPtr<IOMapper>  copyMapperForDeviceWithIndex(IOService * device, unsigned int index);

// { subclasses

	virtual uint64_t getPageSize(void) const = 0;

	virtual IOReturn iovmMapMemory(IOMemoryDescriptor          * memory,
	    uint64_t                      descriptorOffset,
	    uint64_t                      length,
	    uint32_t                      mapOptions,
	    const IODMAMapSpecification * mapSpecification,
	    IODMACommand                * dmaCommand,
	    const IODMAMapPageList      * pageList,
	    uint64_t                    * mapAddress,
	    uint64_t                    * mapLength) = 0;

	virtual IOReturn iovmUnmapMemory(IOMemoryDescriptor * memory,
	    IODMACommand       * dmaCommand,
	    uint64_t             mapAddress,
	    uint64_t             mapLength) = 0;

	virtual IOReturn iovmInsert(uint32_t options,
	    uint64_t mapAddress,
	    uint64_t offset,
	    uint64_t physicalAddress,
	    uint64_t length) = 0;

	virtual uint64_t mapToPhysicalAddress(uint64_t mappedAddress) = 0;

// }

private:
	OSMetaClassDeclareReservedUnused(IOMapper, 0);
	OSMetaClassDeclareReservedUnused(IOMapper, 1);
	OSMetaClassDeclareReservedUnused(IOMapper, 2);
	OSMetaClassDeclareReservedUnused(IOMapper, 3);
	OSMetaClassDeclareReservedUnused(IOMapper, 4);
	OSMetaClassDeclareReservedUnused(IOMapper, 5);
	OSMetaClassDeclareReservedUnused(IOMapper, 6);
	OSMetaClassDeclareReservedUnused(IOMapper, 7);
	OSMetaClassDeclareReservedUnused(IOMapper, 8);
	OSMetaClassDeclareReservedUnused(IOMapper, 9);
	OSMetaClassDeclareReservedUnused(IOMapper, 10);
	OSMetaClassDeclareReservedUnused(IOMapper, 11);
	OSMetaClassDeclareReservedUnused(IOMapper, 12);
	OSMetaClassDeclareReservedUnused(IOMapper, 13);
	OSMetaClassDeclareReservedUnused(IOMapper, 14);
	OSMetaClassDeclareReservedUnused(IOMapper, 15);
};

#endif /* __cplusplus */

#endif /* !__IOKIT_IOMAPPER_H */
