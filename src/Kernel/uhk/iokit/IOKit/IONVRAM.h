/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#ifndef _IOKIT_IONVRAM_H
#define _IOKIT_IONVRAM_H

#ifdef __cplusplus
#include <libkern/c++/OSPtr.h>
#include <IOKit/IOKitKeys.h>
#include <IOKit/IOService.h>
#include <IOKit/IODeviceTreeSupport.h>
#include <IOKit/nvram/IONVRAMController.h>
#endif /* __cplusplus */
#include <uuid/uuid.h>

enum NVRAMPartitionType {
	kIONVRAMPartitionTypeUnknown,
	kIONVRAMPartitionSystem,
	kIONVRAMPartitionCommon
};

enum IONVRAMVariableType {
	kOFVariableTypeBoolean = 1,
	kOFVariableTypeNumber,
	kOFVariableTypeString,
	kOFVariableTypeData
};

enum IONVRAMOperation {
	kIONVRAMOperationInit,
	kIONVRAMOperationRead,
	kIONVRAMOperationWrite,
	kIONVRAMOperationDelete,
	kIONVRAMOperationObliterate,
	kIONVRAMOperationReset
};

enum {
	// Deprecated but still used in AppleEFIRuntime for now
	kOFVariablePermRootOnly = 0,
	kOFVariablePermUserRead,
	kOFVariablePermUserWrite,
	kOFVariablePermKernelOnly
};

#ifdef __cplusplus

class IODTNVRAMVariables;
class IODTNVRAMDiags;
class IODTNVRAMPlatformNotifier;
class IODTNVRAMFormatHandler;

class IODTNVRAM : public IOService
{
	OSDeclareDefaultStructors(IODTNVRAM);

private:
	friend class IODTNVRAMVariables;
	friend class IONVRAMCHRPHandler;
	friend class IONVRAMV3Handler;

	IODTNVRAMPlatformNotifier *_notifier;
	IODTNVRAMDiags            *_diags;
	IODTNVRAMFormatHandler    *_format;

	IODTNVRAMVariables     *_commonService;
	IODTNVRAMVariables     *_systemService;

	SInt32                 _lastDeviceSync;
	bool                   _freshInterval;
	bool                   x86Device = true;

	void initImageFormat(void);

	uint32_t getNVRAMSize(void);

	IOReturn flushGUID(const uuid_t guid, IONVRAMOperation op);
	bool handleSpecialVariables(const char *name, const uuid_t guid, const OSObject *obj, IOReturn *error);

	IOReturn setPropertyInternal(const OSSymbol *aKey, OSObject *anObject);
	IOReturn removePropertyInternal(const OSSymbol *aKey);
	OSSharedPtr<OSObject> copyPropertyWithGUIDAndName(const uuid_t guid, const char *name) const;
	IOReturn removePropertyWithGUIDAndName(const uuid_t guid, const char *name);
	IOReturn setPropertyWithGUIDAndName(const uuid_t guid, const char *name, OSObject *anObject);

	IOReturn syncInternal(bool rateLimit);
	bool safeToSync(void);

public:
	virtual bool init(IORegistryEntry *old, const IORegistryPlane *plane) APPLE_KEXT_OVERRIDE;
	virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;

	virtual void registerNVRAMController(IONVRAMController *controller);

	virtual IOReturn sync(void);
	virtual void reload(void);
	virtual IOReturn getVarDict(OSSharedPtr<OSDictionary> &varDictCopy);
	virtual bool serializeProperties(OSSerialize *s) const APPLE_KEXT_OVERRIDE;
	virtual OSPtr<OSDictionary> dictionaryWithProperties(void) const APPLE_KEXT_OVERRIDE;
	virtual OSPtr<OSObject> copyProperty(const OSSymbol *aKey) const APPLE_KEXT_OVERRIDE;
	virtual OSPtr<OSObject> copyProperty(const char *aKey) const APPLE_KEXT_OVERRIDE;
	virtual OSObject *getProperty(const OSSymbol *aKey) const APPLE_KEXT_OVERRIDE;
	virtual OSObject *getProperty(const char *aKey) const APPLE_KEXT_OVERRIDE;
	virtual bool setProperty(const OSSymbol *aKey, OSObject *anObject) APPLE_KEXT_OVERRIDE;
	virtual void removeProperty(const OSSymbol *aKey) APPLE_KEXT_OVERRIDE;
	virtual IOReturn setProperties(OSObject *properties) APPLE_KEXT_OVERRIDE;

	virtual IOReturn readXPRAM(IOByteCount offset, uint8_t *buffer,
	    IOByteCount length);
	virtual IOReturn writeXPRAM(IOByteCount offset, uint8_t *buffer,
	    IOByteCount length);

	virtual IOReturn readNVRAMProperty(IORegistryEntry *entry,
	    const OSSymbol **name,
	    OSData **value);
	virtual IOReturn writeNVRAMProperty(IORegistryEntry *entry,
	    const OSSymbol *name,
	    OSData *value);

	virtual OSDictionary *getNVRAMPartitions(void);

	virtual IOReturn readNVRAMPartition(const OSSymbol *partitionID,
	    IOByteCount offset, uint8_t *buffer,
	    IOByteCount length);

	virtual IOReturn writeNVRAMPartition(const OSSymbol *partitionID,
	    IOByteCount offset, uint8_t *buffer,
	    IOByteCount length);

	virtual IOByteCount savePanicInfo(uint8_t *buffer, IOByteCount length);
};

#endif /* __cplusplus */

#endif /* !_IOKIT_IONVRAM_H */
