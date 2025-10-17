/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#ifndef __AppleSmartBatteryManagerUserClient__
#define __AppleSmartBatteryManagerUserClient__

#include <IOKit/IOUserClient.h>
#include <IOKit/pwr_mgt/IOPMPowerSource.h>
#include "AppleSmartBatteryManager.h"

/*
 * Method index
 */
enum {
    kSBInflowDisable        = 0,
    kSBChargeInhibit        = 1,
    kSBSetPollingInterval   = 2,
    kSBSMBusReadWriteWord   = 3,
    kSBRequestPoll          = 4,
    kSBSetOverrideCapacity  = 5,
    kSBSwitchToTrueCapacity = 6
};

#define kNumBattMethods     7

/*
 * user client types
 */
enum {
    kSBDefaultType = 0,
    kSBExclusiveSMBusAccessType = 1
};

class AppleSmartBatteryManager;

class AppleSmartBatteryManagerUserClient : public IOUserClient
{
    OSDeclareDefaultStructors(AppleSmartBatteryManagerUserClient)

    friend class AppleSmartBatteryManager;

private:
    AppleSmartBatteryManager *      fOwner;
    task_t                          fOwningTask;
    uint8_t                         fUserClientType;

    IOReturn    secureInflowDisable(int level, int *return_code);
    IOReturn    secureChargeInhibit(int level, int *return_code);
    IOReturn    clientCloseGated(void);
public:

    virtual IOReturn clientClose( void ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn externalMethod(uint32_t selector,
                                IOExternalMethodArguments * arguments,
                                IOExternalMethodDispatch * dispatch = 0,
                                OSObject * targe    = 0, void * reference = 0 ) APPLE_KEXT_OVERRIDE;

    virtual bool start( IOService * provider ) APPLE_KEXT_OVERRIDE;

    virtual bool initWithTask(task_t owningTask, void *security_id, 
                    UInt32 type, OSDictionary * properties) APPLE_KEXT_OVERRIDE;
};

#endif /* ! __AppleSmartBatteryManagerUserClient__ */

