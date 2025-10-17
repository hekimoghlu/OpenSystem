/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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
 * HISTORY
 *
 */


#ifndef _IOKIT_ROOTDOMAINUSERCLIENT_H
#define _IOKIT_ROOTDOMAINUSERCLIENT_H

#include <IOKit/IOUserClient.h>
#include <IOKit/pwr_mgt/IOPM.h>
#include <IOKit/pwr_mgt/RootDomain.h>
#include <IOKit/pwr_mgt/IOPMLibDefs.h>


class RootDomainUserClient : public IOUserClient2022
{
	OSDeclareDefaultStructors(RootDomainUserClient);

	friend class IOPMrootDomain;
private:
	IOPMrootDomain *    fOwner;
	task_t              fOwningTask;

	IOReturn            secureSleepSystem( uint32_t *return_code );

	IOReturn            secureSleepSystemOptions( const void  *inOptions,
	    IOByteCount  inOptionsSize,
	    uint32_t  *returnCode);

	IOReturn            secureSetAggressiveness( unsigned long type,
	    unsigned long newLevel,
	    int *return_code );

	IOReturn            secureSetMaintenanceWakeCalendar(
		IOPMCalendarStruct  *inCalendar,
		uint32_t            *returnCode);

	IOReturn            secureSetUserAssertionLevels(uint32_t    assertionBitfield);

	IOReturn            secureGetSystemSleepType( uint32_t *sleepType, uint32_t *sleepTimer);

	IOReturn            secureAttemptIdleSleepAbort( uint32_t *outReverted);

public:

	virtual IOReturn clientClose( void ) APPLE_KEXT_OVERRIDE;

	virtual IOReturn externalMethod(uint32_t selector,
	    IOExternalMethodArgumentsOpaque * args) APPLE_KEXT_OVERRIDE;

	static IOReturn externalMethodDispatched(OSObject * target, void * reference, IOExternalMethodArguments * args);

	virtual bool start( IOService * provider ) APPLE_KEXT_OVERRIDE;

	virtual bool initWithTask(task_t owningTask, void *security_id,
	    UInt32 type, OSDictionary * properties) APPLE_KEXT_OVERRIDE;

// Unused - retained for symbol compatibility
	void setPreventative(UInt32 on_off, UInt32 types_of_sleep);

// Unused - retained for symbol compatibility
	virtual IOExternalMethod * getTargetAndMethodForIndex( IOService ** targetP, UInt32 index ) APPLE_KEXT_OVERRIDE;
	virtual void stop( IOService *provider) APPLE_KEXT_OVERRIDE;
};

#endif /* ! _IOKIT_ROOTDOMAINUSERCLIENT_H */
