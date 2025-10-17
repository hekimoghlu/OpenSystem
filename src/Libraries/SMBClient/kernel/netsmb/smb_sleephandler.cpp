/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include <IOKit/IOMessage.h>
#include <IOKit/pwr_mgt/RootDomain.h>

extern "C"
{
#include <sys/smb_apple.h>
#include <sys/syslog.h>
#include <sys/kernel.h>
#include <netsmb/smb_subr.h>

	int32_t gSMBSleeping = 0;
	struct timespec gWakeTime = {0, 0};
    void wakeup(void *);
}
#include <netsmb/smb_sleephandler.h>


static IOReturn
smb_sleepwakehandler(void *target, void *refCon, UInt32 messageType, IOService *provider, void *messageArgument, vm_size_t argSize)
{
#pragma unused (target, refCon, provider, messageArgument, argSize)
	switch (messageType) {

	case kIOMessageSystemWillSleep:
		SMBDEBUG(" going to sleep\n");
		gSMBSleeping = 1;
		break;

	case kIOMessageSystemHasPoweredOn:
		SMBDEBUG("  waking up\n");
		gSMBSleeping = 0;
		nanouptime(&gWakeTime);
		break;
        
	default:
		break;
	}
    
	return (IOPMAckImplied);
}

extern "C" {
	IONotifier *fNotifier = NULL;

	__attribute__((visibility("hidden"))) void smbfs_install_sleep_wake_notifier()
	{
		fNotifier = registerSleepWakeInterest(smb_sleepwakehandler, NULL, NULL);
	}

	__attribute__((visibility("hidden"))) void smbfs_remove_sleep_wake_notifier()
	{
		if (fNotifier != NULL) {
			fNotifier->disable();
			//fNotifier->release();  /* if you call this, you kernel panic radar 2946001 */
			fNotifier = NULL;
		}
	}
}
