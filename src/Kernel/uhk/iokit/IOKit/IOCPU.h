/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
 * Copyright (c) 1999 Apple Computer, Inc.  All rights reserved.
 *
 *  DRI: Josh de Cesare
 *
 */

#ifndef _IOKIT_CPU_H
#define _IOKIT_CPU_H

extern "C" {
#include <pexpert/pexpert.h>
}

#include <machine/machine_routines.h>
#include <IOKit/IOService.h>
#include <IOKit/IOInterruptController.h>
#include <IOKit/IOPlatformActions.h>
#include <libkern/c++/OSPtr.h>

enum {
	kIOCPUStateUnregistered = 0,
	kIOCPUStateUninitalized,
	kIOCPUStateStopped,
	kIOCPUStateRunning,
	kIOCPUStateCount
};

class IOCPU : public IOService
{
	OSDeclareAbstractStructors(IOCPU);

private:
	OSPtr<OSArray> _cpuGroup;
	UInt32                 _cpuNumber;
	UInt32                 _cpuState;

protected:
	IOService              *cpuNub;
	processor_t            machProcessor;
	ipi_handler_t          ipi_handler;

	struct ExpansionData { };
	ExpansionData *iocpu_reserved;

	virtual void           setCPUNumber(UInt32 cpuNumber);
	virtual void           setCPUState(UInt32 cpuState);

public:
	virtual bool           start(IOService *provider) APPLE_KEXT_OVERRIDE;
	virtual void           detach(IOService *provider) APPLE_KEXT_OVERRIDE;

	virtual OSObject       *getProperty(const OSSymbol *aKey) const APPLE_KEXT_OVERRIDE;
	virtual bool           setProperty(const OSSymbol *aKey, OSObject *anObject) APPLE_KEXT_OVERRIDE;
	virtual bool           serializeProperties(OSSerialize *serialize) const APPLE_KEXT_OVERRIDE;
	virtual IOReturn       setProperties(OSObject *properties) APPLE_KEXT_OVERRIDE;
	virtual void           initCPU(bool boot) = 0;
	virtual void           quiesceCPU(void) = 0;
	virtual kern_return_t  startCPU(vm_offset_t start_paddr,
	    vm_offset_t arg_paddr) = 0;
	virtual void           haltCPU(void) = 0;
	virtual void           signalCPU(IOCPU *target);
	virtual void           signalCPUDeferred(IOCPU * target);
	virtual void           signalCPUCancel(IOCPU * target);
	virtual void           enableCPUTimeBase(bool enable);

	virtual UInt32         getCPUNumber(void);
	virtual UInt32         getCPUState(void);
	virtual OSArray        *getCPUGroup(void);
	virtual UInt32         getCPUGroupSize(void);
	virtual processor_t    getMachProcessor(void);

	virtual const OSSymbol *getCPUName(void) = 0;

	OSMetaClassDeclareReservedUnused(IOCPU, 0);
	OSMetaClassDeclareReservedUnused(IOCPU, 1);
	OSMetaClassDeclareReservedUnused(IOCPU, 2);
	OSMetaClassDeclareReservedUnused(IOCPU, 3);
	OSMetaClassDeclareReservedUnused(IOCPU, 4);
	OSMetaClassDeclareReservedUnused(IOCPU, 5);
	OSMetaClassDeclareReservedUnused(IOCPU, 6);
	OSMetaClassDeclareReservedUnused(IOCPU, 7);
};

class IOCPUInterruptController : public IOInterruptController
{
	OSDeclareDefaultStructors(IOCPUInterruptController);

private:
	int   enabledCPUs;

protected:
	int   numCPUs;
	int   numSources;

	struct ExpansionData { };
	ExpansionData *iocpuic_reserved;

public:
	virtual IOReturn initCPUInterruptController(int sources);
	virtual void     registerCPUInterruptController(void);
	virtual void     enableCPUInterrupt(IOCPU *cpu);

	virtual void     setCPUInterruptProperties(IOService *service) APPLE_KEXT_OVERRIDE;
	virtual IOReturn registerInterrupt(IOService *nub, int source,
	    void *target,
	    IOInterruptHandler handler,
	    void *refCon) APPLE_KEXT_OVERRIDE;

	virtual IOReturn getInterruptType(IOService *nub, int source,
	    int *interruptType) APPLE_KEXT_OVERRIDE;

	virtual IOReturn enableInterrupt(IOService *nub, int source) APPLE_KEXT_OVERRIDE;
	virtual IOReturn disableInterrupt(IOService *nub, int source) APPLE_KEXT_OVERRIDE;
	virtual IOReturn causeInterrupt(IOService *nub, int source) APPLE_KEXT_OVERRIDE;

	virtual IOReturn handleInterrupt(void *refCon, IOService *nub,
	    int source) APPLE_KEXT_OVERRIDE;

	virtual IOReturn initCPUInterruptController(int sources, int cpus);

	OSMetaClassDeclareReservedUnused(IOCPUInterruptController, 1);
	OSMetaClassDeclareReservedUnused(IOCPUInterruptController, 2);
	OSMetaClassDeclareReservedUnused(IOCPUInterruptController, 3);
	OSMetaClassDeclareReservedUnused(IOCPUInterruptController, 4);
	OSMetaClassDeclareReservedUnused(IOCPUInterruptController, 5);
};

#endif /* ! _IOKIT_CPU_H */
