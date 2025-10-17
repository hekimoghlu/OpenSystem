/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
extern "C" {
#include <kern/debug.h>
#include <kern/queue.h>
}

#include <kern/sched_prim.h>
#include <machine/machine_routines.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOPlatformExpert.h>
#include <IOKit/IOKitKeysPrivate.h>
#include <IOKit/IOPlatformActions.h>
#include "IOKitKernelInternal.h"

static IOLock *gIOPlatformActionsLock;

typedef kern_return_t (*iocpu_platform_action_t)(void * refcon0, void * refcon1, uint32_t priority,
    void * param1, void * param2, void * param3,
    const char * name, uint64_t platform_action_flags);

struct iocpu_platform_action_entry {
	queue_chain_t                     link;
	iocpu_platform_action_t           action;
	int32_t                           priority;
	const char *                      name;
	void *                            refcon0;
	void *                            refcon1;
	boolean_t                         callout_in_progress;
	struct iocpu_platform_action_entry * alloc_list;
};
typedef struct iocpu_platform_action_entry iocpu_platform_action_entry_t;

enum {
	kQueueSleep       = 0,
	kQueueWake        = 1,
	kQueueQuiesce     = 2,
	kQueueActive      = 3,
	kQueueHaltRestart = 4,
	kQueuePanic       = 5,
	kQueueCount       = 6
};

#define PLATFORM_ACTION_FLAGS_ALLOW_NESTED_CALLOUTS 1
#define PLATFORM_ACTION_FLAGS_NO_LOGGING            2

const OSSymbol *                gIOPlatformSleepActionKey;
const OSSymbol *                gIOPlatformWakeActionKey;
const OSSymbol *                gIOPlatformQuiesceActionKey;
const OSSymbol *                gIOPlatformActiveActionKey;
const OSSymbol *                gIOPlatformHaltRestartActionKey;
const OSSymbol *                gIOPlatformPanicActionKey;

static queue_head_t             gActionQueues[kQueueCount];
static const OSSymbol *         gActionSymbols[kQueueCount];

static bool
IOInstallServicePlatformAction(IOService * service, uint32_t qidx);

static void
iocpu_add_platform_action(queue_head_t * queue, iocpu_platform_action_entry_t * entry)
{
	iocpu_platform_action_entry_t * next;

	queue_iterate(queue, next, iocpu_platform_action_entry_t *, link)
	{
		if (next->priority > entry->priority) {
			queue_insert_before(queue, entry, next, iocpu_platform_action_entry_t *, link);
			return;
		}
	}
	queue_enter(queue, entry, iocpu_platform_action_entry_t *, link); // at tail
}

static void
iocpu_remove_platform_action(iocpu_platform_action_entry_t * entry)
{
	remque(&entry->link);
}

static kern_return_t
iocpu_run_platform_actions(queue_head_t * queue, uint32_t first_priority, uint32_t last_priority,
    void * param1, void * param2, void * param3, uint64_t platform_action_flags)
{
	kern_return_t                ret = KERN_SUCCESS;
	kern_return_t                result = KERN_SUCCESS;
	iocpu_platform_action_entry_t * next;
	boolean_t allow_nested_callouts = (platform_action_flags & PLATFORM_ACTION_FLAGS_ALLOW_NESTED_CALLOUTS);

	queue_iterate(queue, next, iocpu_platform_action_entry_t *, link)
	{
		uint32_t pri = (next->priority < 0) ? -next->priority : next->priority;
		if ((pri >= first_priority) && (pri <= last_priority)) {
			if (!allow_nested_callouts && !next->callout_in_progress) {
				next->callout_in_progress = TRUE;
				ret = (*next->action)(next->refcon0, next->refcon1, pri, param1, param2, param3, next->name, platform_action_flags);
				next->callout_in_progress = FALSE;
			} else if (allow_nested_callouts) {
				ret = (*next->action)(next->refcon0, next->refcon1, pri, param1, param2, param3, next->name, platform_action_flags);
			}
		}
		if (KERN_SUCCESS == result) {
			result = ret;
		}
	}
	return result;
}

extern "C" kern_return_t
IOCPURunPlatformQuiesceActions(void)
{
	assert(preemption_enabled() == false);
	return iocpu_run_platform_actions(&gActionQueues[kQueueQuiesce], 0, 0U - 1,
	           NULL, NULL, NULL, PLATFORM_ACTION_FLAGS_ALLOW_NESTED_CALLOUTS);
}

extern "C" kern_return_t
IOCPURunPlatformActiveActions(void)
{
	assert(preemption_enabled() == false);
	ml_hibernate_active_pre();
	kern_return_t result = iocpu_run_platform_actions(&gActionQueues[kQueueActive], 0, 0U - 1,
	    NULL, NULL, NULL, PLATFORM_ACTION_FLAGS_ALLOW_NESTED_CALLOUTS);
	ml_hibernate_active_post();
	return result;
}

extern "C" kern_return_t
IOCPURunPlatformHaltRestartActions(uint32_t message)
{
	if (!gActionQueues[kQueueHaltRestart].next) {
		return kIOReturnNotReady;
	}
	return iocpu_run_platform_actions(&gActionQueues[kQueueHaltRestart], 0, 0U - 1,
	           (void *)(uintptr_t) message, NULL, NULL, PLATFORM_ACTION_FLAGS_ALLOW_NESTED_CALLOUTS);
}

extern "C" kern_return_t
IOCPURunPlatformPanicActions(uint32_t message, uint32_t details)
{
	// Don't allow nested calls of panic actions
	if (!gActionQueues[kQueuePanic].next) {
		return kIOReturnNotReady;
	}
	uint64_t platform_action_flags = 0;

	if (!verbose_panic_flow_logging) {
		platform_action_flags = PLATFORM_ACTION_FLAGS_NO_LOGGING;
	}
	return iocpu_run_platform_actions(&gActionQueues[kQueuePanic], 0, 0U - 1,
	           (void *)(uintptr_t) message, (void *)(uintptr_t) details, NULL, platform_action_flags);
}

extern "C" kern_return_t
IOCPURunPlatformPanicSyncAction(void *addr, uint32_t offset, uint32_t len)
{
	PE_panic_save_context_t context = {
		.psc_buffer = addr,
		.psc_offset = offset,
		.psc_length = len
	};

	// Don't allow nested calls of panic actions
	if (!gActionQueues[kQueuePanic].next) {
		return kIOReturnNotReady;
	}
	return iocpu_run_platform_actions(&gActionQueues[kQueuePanic], 0, 0U - 1,
	           (void *)(uintptr_t)(kPEPanicSync), &context, NULL, FALSE);
}

void
IOPlatformActionsPreSleep(void)
{
	iocpu_run_platform_actions(&gActionQueues[kQueueSleep], 0, 0U - 1,
	    NULL, NULL, NULL, PLATFORM_ACTION_FLAGS_ALLOW_NESTED_CALLOUTS);
}

void
IOPlatformActionsPostResume(void)
{
	iocpu_run_platform_actions(&gActionQueues[kQueueWake], 0, 0U - 1,
	    NULL, NULL, NULL, PLATFORM_ACTION_FLAGS_ALLOW_NESTED_CALLOUTS);
}

void
IOPlatformActionsInitialize(void)
{
	gIOPlatformActionsLock = IOLockAlloc();

	for (uint32_t qidx = kQueueSleep; qidx < kQueueCount; qidx++) {
		queue_init(&gActionQueues[qidx]);
	}

	gIOPlatformSleepActionKey        = gActionSymbols[kQueueSleep]
	            = OSSymbol::withCStringNoCopy(kIOPlatformSleepActionKey);
	gIOPlatformWakeActionKey         = gActionSymbols[kQueueWake]
	            = OSSymbol::withCStringNoCopy(kIOPlatformWakeActionKey);
	gIOPlatformQuiesceActionKey      = gActionSymbols[kQueueQuiesce]
	            = OSSymbol::withCStringNoCopy(kIOPlatformQuiesceActionKey);
	gIOPlatformActiveActionKey       = gActionSymbols[kQueueActive]
	            = OSSymbol::withCStringNoCopy(kIOPlatformActiveActionKey);
	gIOPlatformHaltRestartActionKey  = gActionSymbols[kQueueHaltRestart]
	            = OSSymbol::withCStringNoCopy(kIOPlatformHaltRestartActionKey);
	gIOPlatformPanicActionKey = gActionSymbols[kQueuePanic]
	            = OSSymbol::withCStringNoCopy(kIOPlatformPanicActionKey);
}

static kern_return_t
IOServicePlatformAction(void * refcon0, void * refcon1, uint32_t priority,
    void * param1, void * param2, void * param3,
    const char * service_name, uint64_t platform_action_flags)
{
	IOReturn         ret;
	IOService *      service  = (IOService *)      refcon0;
	const OSSymbol * function = (const OSSymbol *) refcon1;

	if (!(platform_action_flags & PLATFORM_ACTION_FLAGS_NO_LOGGING)) {
		IOLog("%s -> %s\n", function->getCStringNoCopy(), service_name);
	}

	/*
	 * We intentionally don't trace params that are kernel addresses,
	 * and truncate 64 bit values to 32 bit, so they all fit into
	 * one tracepoint along with IOService registry id.
	 */
	SOCD_TRACE_XNU_START(PLATFORM_ACTION,
	    ADDR(function->getCStringNoCopy()),
	    ADDR(service->getMetaClass()),
	    PACK_2X32(VALUE(param1), VALUE(service->getRegistryEntryID())),
	    PACK_2X32(VALUE(param3), VALUE(param2)));

	ret = service->callPlatformFunction(function, false,
	    (void *)(uintptr_t) priority, param1, param2, param3);

	SOCD_TRACE_XNU_END(PLATFORM_ACTION,
	    ADDR(function->getCStringNoCopy()),
	    ADDR(service->getMetaClass()),
	    PACK_2X32(VALUE(param1), VALUE(service->getRegistryEntryID())),
	    PACK_2X32(VALUE(param3), VALUE(param2)));

	return ret;
}

static bool
IOInstallServicePlatformAction(IOService * service, uint32_t qidx)
{
	iocpu_platform_action_entry_t * entry;
	OSNumber *       num;
	uint32_t         priority;
	const OSSymbol * key = gActionSymbols[qidx];
	queue_head_t *   queue = &gActionQueues[qidx];
	bool             reverse;

	num = OSDynamicCast(OSNumber, service->getProperty(key));
	if (!num) {
		return true;
	}

	reverse = false;
	switch (qidx) {
	case kQueueWake:
	case kQueueActive:
		reverse = true;
		break;
	}
	queue_iterate(queue, entry, iocpu_platform_action_entry_t *, link)
	{
		if (service == entry->refcon0) {
			return true;
		}
	}

	entry = IOMallocType(iocpu_platform_action_entry_t);
	entry->action = &IOServicePlatformAction;
	entry->name = service->getName();
	priority = num->unsigned32BitValue();
	if (reverse) {
		entry->priority = -priority;
	} else {
		entry->priority = priority;
	}
	entry->refcon0 = service;
	entry->refcon1 = (void *) key;
	entry->callout_in_progress = FALSE;

	iocpu_add_platform_action(queue, entry);
	return false;
}


IOReturn
IOInstallServicePlatformActions(IOService * service)
{
	IOLockLock(gIOPlatformActionsLock);

	IOInstallServicePlatformAction(service, kQueueHaltRestart);
	IOInstallServicePlatformAction(service, kQueuePanic);

	IOLockUnlock(gIOPlatformActionsLock);

	return kIOReturnSuccess;
}

IOReturn
IOInstallServiceSleepPlatformActions(IOService * service)
{
	IOLockLock(gIOPlatformActionsLock);

	for (uint32_t qidx = kQueueSleep; qidx <= kQueueActive; qidx++) {
		IOInstallServicePlatformAction(service, qidx);
	}

	IOLockUnlock(gIOPlatformActionsLock);

	return kIOReturnSuccess;
}

IOReturn
IORemoveServicePlatformActions(IOService * service)
{
	iocpu_platform_action_entry_t * entry;
	iocpu_platform_action_entry_t * next;

	IOLockLock(gIOPlatformActionsLock);

	for (uint32_t qidx = kQueueSleep; qidx < kQueueCount; qidx++) {
		next = (typeof(entry))queue_first(&gActionQueues[qidx]);
		while (!queue_end(&gActionQueues[qidx], &next->link)) {
			entry = next;
			next = (typeof(entry))queue_next(&entry->link);
			if (service == entry->refcon0) {
				iocpu_remove_platform_action(entry);
				IOFreeType(entry, iocpu_platform_action_entry_t);
			}
		}
	}

	IOLockUnlock(gIOPlatformActionsLock);

	return kIOReturnSuccess;
}
