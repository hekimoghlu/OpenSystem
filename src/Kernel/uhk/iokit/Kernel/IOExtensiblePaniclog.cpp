/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#include <IOKit/IOExtensiblePaniclog.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOBSD.h>
#include <IOKit/IOBufferMemoryDescriptor.h>

#include <libkern/c++/OSAllocation.h>
#include <libkern/c++/OSKext.h>

__BEGIN_DECLS
#include <os/log.h>
__END_DECLS

#define super OSObject
OSDefineMetaClassAndStructors(IOExtensiblePaniclog, OSObject)

bool
IOExtensiblePaniclog::init(void)
{
	extPaniclogHandle = NULL;

	if (!super::init()) {
		os_log_error(OS_LOG_DEFAULT, "EXT_PANICLOG: Super init failed\n");
		return false;
	}

	return true;
}

bool
IOExtensiblePaniclog::createWithUUID(uuid_t uuid, const char *data_id, uint32_t len,
    ext_paniclog_create_options_t options, IOExtensiblePaniclog **out)
{
	IOExtensiblePaniclog *inst = OSTypeAlloc(IOExtensiblePaniclog);
	if (!inst) {
		os_log_error(OS_LOG_DEFAULT, "EXT_PANICLOG: instance is NULL\n");
		return false;
	}

	if (!inst->init()) {
		os_log_error(OS_LOG_DEFAULT, "EXT_PANICLOG: init failed\n");
		OSSafeReleaseNULL(inst);
		return false;
	}

	inst->extPaniclogHandle = ext_paniclog_handle_alloc_with_uuid(uuid, data_id,
	    len, options);
	if (inst->extPaniclogHandle == NULL) {
		os_log(OS_LOG_DEFAULT, "EXT_PANICLOG: Handle alloc failed\n");
		OSSafeReleaseNULL(inst);
		return false;
	}

	*out = inst;

	return true;
}

void
IOExtensiblePaniclog::free(void)
{
	if (extPaniclogHandle != NULL) {
		ext_paniclog_handle_free(extPaniclogHandle);
	}

	if (iomd != NULL) {
		iomd->release();
	}

	super::free();
}

int
IOExtensiblePaniclog::setActive()
{
	return ext_paniclog_handle_set_active(extPaniclogHandle);
}

int
IOExtensiblePaniclog::setInactive()
{
	return ext_paniclog_handle_set_inactive(extPaniclogHandle);
}

int
IOExtensiblePaniclog::insertData(void *addr, uint32_t len)
{
	return ext_paniclog_insert_data(extPaniclogHandle, addr, len);
}

int
IOExtensiblePaniclog::appendData(void *addr, uint32_t len)
{
	return ext_paniclog_append_data(extPaniclogHandle, addr, len);
}

void *
IOExtensiblePaniclog::claimBuffer()
{
	return ext_paniclog_claim_buffer(extPaniclogHandle);
}

int
IOExtensiblePaniclog::yieldBuffer(uint32_t used_len)
{
	return ext_paniclog_yield_buffer(extPaniclogHandle, used_len);
}

int
IOExtensiblePaniclog::setUsedLen(uint32_t used_len)
{
	return ext_paniclog_set_used_len(extPaniclogHandle, used_len);
}

/*********************************************************************************
*                                                                               *
*  Driver Kit functions                                                         *
*                                                                               *
*********************************************************************************/

kern_return_t
IOExtensiblePaniclog::Create_Impl(OSData *uuid, OSString *data_id, uint32_t max_len,
    uint32_t options, IOExtensiblePaniclog **out)
{
	IOExtensiblePaniclog * inst = NULL;
	uuid_t uuid_copy;
	uint32_t mem_options = 0;

	if (!IOCurrentTaskHasEntitlement(EXTPANICLOG_ENTITLEMENT)) {
		return kIOReturnNotPrivileged;
	}

	if ((uuid == nullptr) || (uuid->getLength() > sizeof(uuid_t))) {
		return kIOReturnBadArgument;
	}

	if ((data_id == nullptr) || (data_id->getLength() > MAX_DATA_ID_SIZE)) {
		return kIOReturnBadArgument;
	}

	memcpy(&uuid_copy, uuid->getBytesNoCopy(), uuid->getLength());

	inst = OSTypeAlloc(IOExtensiblePaniclog);
	if (!inst->init()) {
		OSSafeReleaseNULL(inst);
		return kIOReturnNoMemory;
	}

	mem_options = kIOMemoryKernelUserShared | kIOMemoryThreadSafe | kIODirectionInOut;
	inst->iomd = IOBufferMemoryDescriptor::withOptions(mem_options, max_len);
	if (inst->iomd == NULL) {
		IOLog("EXT_PANICLOG IOKIT: Failed to create iobmd");
		OSSafeReleaseNULL(inst);
		return kIOReturnNoMemory;
	}

	inst->extPaniclogHandle = ext_paniclog_handle_alloc_with_buffer(uuid_copy,
	    data_id->getCStringNoCopy(), max_len, inst->iomd->getBytesNoCopy(),
	    (ext_paniclog_create_options_t)(options | EXT_PANICLOG_OPTIONS_WITH_BUFFER));
	if (inst->extPaniclogHandle == NULL) {
		OSSafeReleaseNULL(inst);
		return kIOReturnNoMemory;
	}

	*out = inst;

	return kIOReturnSuccess;
}

kern_return_t
IOExtensiblePaniclog::SetActive_Impl()
{
	if (ext_paniclog_handle_set_active(extPaniclogHandle) != 0) {
		return kIOReturnBadArgument;
	}

	return kIOReturnSuccess;
}

kern_return_t
IOExtensiblePaniclog::SetInactive_Impl()
{
	if (ext_paniclog_handle_set_inactive(extPaniclogHandle) != 0) {
		return kIOReturnBadArgument;
	}

	return kIOReturnSuccess;
}

kern_return_t
IOExtensiblePaniclog::InsertData_Impl(OSData *data)
{
	if (data == nullptr) {
		return kIOReturnBadArgument;
	}

	void *addr = (void *)data->getBytesNoCopy();

	if (ext_paniclog_insert_data(extPaniclogHandle, addr, data->getLength()) != 0) {
		return kIOReturnBadArgument;
	}

	return kIOReturnSuccess;
}

kern_return_t
IOExtensiblePaniclog::AppendData_Impl(OSData *data)
{
	if (data == nullptr) {
		return kIOReturnBadArgument;
	}

	void *addr = (void *)data->getBytesNoCopy();

	if (ext_paniclog_append_data(extPaniclogHandle, addr, data->getLength()) != 0) {
		return kIOReturnBadArgument;
	}

	return kIOReturnSuccess;
}

kern_return_t
IOExtensiblePaniclog::CopyMemoryDescriptor_Impl(IOBufferMemoryDescriptor **mem)
{
	(void) ext_paniclog_claim_buffer(extPaniclogHandle);

	iomd->retain();
	*mem = iomd;
	return kIOReturnSuccess;
}

kern_return_t
IOExtensiblePaniclog::SetUsedLen_Impl(uint32_t used_len)
{
	return ext_paniclog_set_used_len(extPaniclogHandle, used_len);
}
