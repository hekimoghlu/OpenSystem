/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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
#include <mach/mach_port.h>

#include <IOKit/kext/OSKext.h>
#include "kext_tools_util.h"
#include "kextd_globals.h"
#include <dispatch/dispatch.h>

dispatch_source_t _gKextutilLock = NULL;

void kextd_process_kernel_requests(void);

/******************************************************************************
 *****************************************************************************/
static void removeKextutilLock(void)
{
    if (_gKextutilLock) {
        dispatch_source_cancel(_gKextutilLock);
    }

    if (gKernelRequestsPending) {
        kextd_process_kernel_requests();
    }

    CFRunLoopWakeUp(CFRunLoopGetCurrent());
    return;
}

/******************************************************************************
 * MIG Server Routine
 * _kextmanager_lock_kextload serializes kextutil clients
 *****************************************************************************/
kern_return_t _kextmanager_lock_kextload(
    mach_port_t server,
    mach_port_t client,
    int * lockstatus)
{
    kern_return_t mig_result = KERN_FAILURE;
    int result = ELAST + 1;

    if (!lockstatus) {
        OSKextLog(/* kext */ NULL,
            kOSKextLogErrorLevel | kOSKextLogIPCFlag,
            "kextmanager_lock_kextload requires non-NULL lockstatus.");
        mig_result = KERN_SUCCESS;
        result = EINVAL;
        goto finish;
    }

    if (gClientUID != 0) {
        OSKextLog(/* kext */ NULL,
            kOSKextLogErrorLevel,
            "Non-root process doesn't need to lock as it will fail to load.");
        mig_result = KERN_SUCCESS;
        result = EPERM;
        goto finish;
    }

    if (_gKextutilLock) {
        mig_result = KERN_SUCCESS;
        result = EBUSY;
        goto finish;
    }

    result = ENOMEM;

    _gKextutilLock = dispatch_source_create(DISPATCH_SOURCE_TYPE_MACH_SEND, client,
                        DISPATCH_MACH_SEND_DEAD, dispatch_get_main_queue());

    if (_gKextutilLock) {

        dispatch_source_set_event_handler(_gKextutilLock, ^{
                OSKextLog(/* kext */ NULL,
                    kOSKextLogErrorLevel | kOSKextLogIPCFlag,
                    "Client exited without releasing kextutil lock.");
                removeKextutilLock();
            });

        dispatch_source_set_cancel_handler(_gKextutilLock, ^{
                dispatch_release(_gKextutilLock);
                mach_port_deallocate(mach_task_self(), client);
                _gKextutilLock = NULL;
            });

        dispatch_resume(_gKextutilLock);

        mig_result = KERN_SUCCESS;
        result = 0;
    }

finish:
    if (mig_result == KERN_SUCCESS && result != 0) {
        /*
         * if result == 0, then we're using the 'client' send right in a
         * dispatch source. However, if we did not consume the 'client'
         * send right (result != 0), and we're returning success
         * (mig_result == KERN_SUCCESS), then we need to consume the right
         * because MIG will not.
         */
        mach_port_deallocate(mach_task_self(), client);
    }

    if (mig_result != KERN_SUCCESS) {
        if (gClientUID == 0) {
            OSKextLog(/* kext */ NULL,
                kOSKextLogErrorLevel | kOSKextLogIPCFlag,
                "Trouble while locking for kextutil - %s.",
                safe_mach_error_string(mig_result));
        }
        removeKextutilLock();
    } else if (lockstatus) {
        *lockstatus = result;  // only meaningful if mig_result == KERN_SUCCESS
    }

    return mig_result;
}

/******************************************************************************
 * MIG Server Routine
 * _kextmanager_unlock_kextload unlocks kextutil clients
 *****************************************************************************/
kern_return_t _kextmanager_unlock_kextload(
    mach_port_t server,
    mach_port_t client)
{
    kern_return_t mig_result = KERN_FAILURE;

    if (gClientUID != 0) {
        OSKextLog(/* kext */ NULL,
            kOSKextLogErrorLevel | kOSKextLogIPCFlag,
            "Non-root kextutil doesn't need to lock/unlock.");
        mig_result = KERN_SUCCESS;
        goto finish;
    }

    if (client != (mach_port_t)dispatch_source_get_handle(_gKextutilLock)) {
        OSKextLog(/* kext */ NULL,
            kOSKextLogErrorLevel | kOSKextLogIPCFlag,
            "%d not used to lock for kextutil.", client);
        goto finish;
    }

    removeKextutilLock();

    mig_result = KERN_SUCCESS;

finish:
    if (mig_result == KERN_SUCCESS) {
        /*
         * We don't need the extra send right added by MiG. Per convention,
         * MiG will automatically free this if we return an error, so only
         * explicitly deallocate if the call was successful.
         */
        mach_port_deallocate(mach_task_self(), client);
    }

    return mig_result;
}

