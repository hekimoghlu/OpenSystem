/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
#include <netinet6/ipsec.h>
#include <netkey/key.h>
#include <IOKit/pwr_mgt/IOPM.h>

void *sleep_wake_handle = NULL;

typedef IOReturn (*IOServiceInterestHandler)( void * target, void * refCon,
    UInt32 messageType, void * provider,
    void * messageArgument, vm_size_t argSize );
extern void *registerSleepWakeInterest(IOServiceInterestHandler, void *, void *);

static IOReturn
ipsec_sleep_wake_handler(void *target, void *refCon, UInt32 messageType,
    void *provider, void *messageArgument, vm_size_t argSize)
{
#pragma unused(target, refCon, provider, messageArgument, argSize)
	switch (messageType) {
	case kIOMessageSystemWillSleep:
	{
		ipsec_get_local_ports();
		break;
	}
	default:
		break;
	}

	return IOPMAckImplied;
}

void
ipsec_monitor_sleep_wake(void)
{
	LCK_MTX_ASSERT(sadb_mutex, LCK_MTX_ASSERT_OWNED);

	if (sleep_wake_handle == NULL) {
		sleep_wake_handle = registerSleepWakeInterest(ipsec_sleep_wake_handler,
		    NULL, NULL);
		if (sleep_wake_handle != NULL) {
			ipseclog((LOG_INFO,
			    "ipsec: monitoring sleep wake"));
		}
	}
}
