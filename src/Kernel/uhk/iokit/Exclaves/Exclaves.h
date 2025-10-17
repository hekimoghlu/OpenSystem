/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#ifndef _IOKIT_EXCLAVES_H
#define _IOKIT_EXCLAVES_H

#if CONFIG_EXCLAVES

#include <kern/thread_call.h>
#include <libkern/OSTypes.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus

#include <libkern/c++/OSDictionary.h>
#include <libkern/c++/OSSymbol.h>

/* Global IOExclaveProxyState lookup table */
extern OSDictionary     *gExclaveProxyStates;
extern IORecursiveLock  *gExclaveProxyStateLock;
extern const OSSymbol * gDARTMapperFunctionSetActive;

extern "C" {
#endif /* __cplusplus */

/* Exclave upcall handler arguments */

enum IOExclaveInterruptUpcallType {
	kIOExclaveInterruptUpcallTypeRegister,
	kIOExclaveInterruptUpcallTypeRemove,
	kIOExclaveInterruptUpcallTypeEnable
};

struct IOExclaveInterruptUpcallArgs {
	int index;
	enum IOExclaveInterruptUpcallType type;
	union {
		struct {
			// Register an IOIES with no provider for testing purposes
			bool test_irq;
		} register_args;
		struct {
			bool enable;
		} enable_args;
	} data;
};

enum IOExclaveTimerUpcallType {
	kIOExclaveTimerUpcallTypeRegister,
	kIOExclaveTimerUpcallTypeRemove,
	kIOExclaveTimerUpcallTypeEnable,
	kIOExclaveTimerUpcallTypeSetTimeout,
	kIOExclaveTimerUpcallTypeCancelTimeout
};

struct IOExclaveTimerUpcallArgs {
	uint32_t timer_id;
	enum IOExclaveTimerUpcallType type;
	union {
		struct {
			bool enable;
		} enable_args;
		struct {
			bool clock_continuous;
			AbsoluteTime duration;
			kern_return_t kr;
		} set_timeout_args;
	} data;
};

enum IOExclaveAsyncNotificationUpcallType {
	AsyncNotificationUpcallTypeSignal,
};

struct IOExclaveAsyncNotificationUpcallArgs {
	enum IOExclaveAsyncNotificationUpcallType type;
	uint32_t notificationID;
};

enum IOExclaveMapperOperationUpcallType {
	MapperActivate,
	MapperDeactivate,
};

struct IOExclaveMapperOperationUpcallArgs {
	enum IOExclaveMapperOperationUpcallType type;
	uint32_t mapperIndex;
};

enum IOExclaveANEUpcallType {
	kIOExclaveANEUpcallTypeSetPowerState,
	kIOExclaveANEUpcallTypeWorkSubmit,
	kIOExclaveANEUpcallTypeWorkBegin,
	kIOExclaveANEUpcallTypeWorkEnd,
};

struct IOExclaveANEUpcallArgs {
	enum IOExclaveANEUpcallType type;
	union {
		struct {
			uint32_t desired_state;
		} setpowerstate_args;
		struct {
			uint64_t arg0;
			uint64_t arg1;
			uint64_t arg2;
		} work_args;
	};
};

/*
 * Exclave upcall handlers
 *
 * id is the registry ID of the proxy IOService.
 */
bool IOExclaveInterruptUpcallHandler(uint64_t id, struct IOExclaveInterruptUpcallArgs *args);
bool IOExclaveTimerUpcallHandler(uint64_t id, struct IOExclaveTimerUpcallArgs *args);
bool IOExclaveLockWorkloop(uint64_t id, bool lock);
bool IOExclaveAsyncNotificationUpcallHandler(uint64_t id, struct IOExclaveAsyncNotificationUpcallArgs *args);
bool IOExclaveMapperOperationUpcallHandler(uint64_t id, struct IOExclaveMapperOperationUpcallArgs *args);
bool IOExclaveANEUpcallHandler(uint64_t id, struct IOExclaveANEUpcallArgs *args, bool *result);

/* Test support */

struct IOExclaveTestSignalInterruptParam {
	uint64_t id;
	uint64_t index;
};
void IOExclaveTestSignalInterrupt(thread_call_param_t, thread_call_param_t);

void exclaves_wait_for_cpu_init(void);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* CONFIG_EXCLAVES */

#endif /* ! _IOKIT_EXCLAVES_H */
