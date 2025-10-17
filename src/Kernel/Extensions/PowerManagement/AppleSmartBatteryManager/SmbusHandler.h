/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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

//
//  SmbusHandler.h
//  PowerManagement
//
//  Created by prasadlv on 11/8/16.
//
//

#ifndef SmbusHandler_h
#define SmbusHandler_h

#include <IOKit/acpi/IOACPIPlatformDevice.h>
#include <libkern/c++/OSObject.h>
#include "AppleSmartBatteryCommands.h"
#include "AppleSmartBatteryManager.h"

#define MAX_SMBUS_DATA_SIZE 32

class AppleSmartBatteryManager;


class SmbusHandler : public OSObject
{
    
OSDeclareDefaultStructors(SmbusHandler)

private:
    int         fRetryAttempts;
    int         fExternalTransactionWait;
    
    AppleSmartBatteryManager    *fMgr;
    IOWorkLoop                  *fWorkLoop;

    ASBMgrTransactionCompletion     fCompletion;
    bool                            fFullyDischarged;
    OSObject                        *fTarget;
    void                            *fReference;
    IOSMBusTransaction              fTransaction;
    ASBMgrOpType                    fOpType;
    uint32_t                        fCmdCount;

    IOACPIPlatformDevice            *fACPIProvider;

    void smbusCompletion(void *ref, IOSMBusTransaction *transaction);
    void smbusExternalTransactionCompletion(void *ref, IOSMBusTransaction *transaction);
    IOReturn getErrorCode(IOSMBusStatus status);

public:

    IOReturn initialize ( AppleSmartBatteryManager *mgr );
    uint32_t requiresRetryGetMicroSec(IOSMBusTransaction *transaction);
    IOReturn isTransactionAllowed();
    IOReturn performTransaction(ASBMgrRequest *req, ASBMgrTransactionCompletion completion, OSObject * target, void * reference);

    /*
     * smbusExternalTransaction - Handles smbus transactions received from user clients.
     * This call is blocked until command is completed.
     */
    IOReturn smbusExternalTransaction(void *in, void *out, IOByteCount inSize, IOByteCount *outSize);
    void handleExclusiveAccess(bool exclusive);
    IOReturn inhibitCharging(int level);
    IOReturn disableInflow(int level);


};

#endif /* SmbusHandler_h */
