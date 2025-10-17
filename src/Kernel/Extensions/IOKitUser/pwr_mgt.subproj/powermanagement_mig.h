/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#ifndef	_powermanagement_mig_h_
#define	_powermanagement_mig_h_

#define kPMMIGStringLength                          1024

typedef char * string_t;

enum {
    kIOPMGetValueDWBTSupportOnAC                        = 1,
    kIOPMGetValueDWBTSupportOnBatt                      = 2
};

/* 
 * 'setMode' values in io_pm_set_debug_flags() MIG call
 */
enum {
    kIOPMDebugFlagsSetBits,
    kIOPMDebugFlagsResetBits,
    kIOPMDebugFlagsSetValue
};

/*
 * Arguments to powermanagement.defs MIG call io_pm_assertion_copy_details
 *    parameter "whichData"
 */
enum {
    kIOPMAssertionMIGCopyOneAssertionProperties     = 1,
    kIOPMAssertionMIGCopyAll                        = 2,
    kIOPMAssertionMIGCopyStatus                     = 3,
    kIOPMPowerEventsMIGCopyScheduledEvents          = 4,
    kIOPMPowerEventsMIGCopyRepeatEvents             = 5,
    kIOPMAssertionMIGCopyByType                     = 6,
    kIOPMAssertionMIGCopyInactive                   = 7,
};

/*
 * Arguments to powermanagement.defs MIG call io_pm_assertion_retain_release
 *    parameter "action"
 */
enum {
    kIOPMAssertionMIGDoRetain                       = 1,
    kIOPMAssertionMIGDoRelease                      = -1
};


/*
 * XPC based messaging keys
 */

#define kMsgReturnCode              "returnCode"
#define kClaimSystemWakeEvent       "claimSystemWakeEvent"

#define kUserActivityRegister           "userActivityRegister"
#define kUserActivityTimeoutUpdate      "userActivityTimeout"

#define kUserActivityTimeoutKey     "ActivityTimeout"
#define kUserActivityLevels         "UserActivityLevels"

#define kAssertionCreateMsg         "assertionCreate"
#define kAssertionReleaseMsg        "assertionRelease"

#define kAssertionPropertiesMsg     "assertionProperties"
#define kAssertionCheckMsg          "assertionCheck"
#define kAssertionTimeoutMsg        "assertionTimeout"
#define kAssertionSetStateMsg       "assertionSetState"
#define kAssertionCopyActivityUpdateMsg "assertionCopyActivityUpdate"
#define kAssertionUpdateActivityMsg     "assertionUpdateActivity"
#define kAssertionUpdateActivesMsg      "assertionUpdateActives"


#define kAssertionDetailsKey        "assertionDictonary"
#define kAssertionIdKey             "assertionId"
#define kAssertionReleaseDateKey    "assertioReleaseDate"
#define kAssertionEnTrIntensityKey  "EnTrIntensity"
#define kAssertionCheckTokenKey     "assertionCheckToken"
#define kAssertionCheckCountKey     "assertionCheckCount"
#define kAssertionActivityLogKey    "assertionActivityLog"
#define kAssertionWasCoalesced      "assertionWasCoalesced"
#define kAssertionInitialConnKey    "assertionInitialConnection"
#define kAssertionFeatureSupportKey  "assertionFeatureSupported"
#define kAssertionAsyncOffloadDelay "assertionAsyncOffloadDelay"
#define kAssertionCopyActivityUpdateRefCntKey "assertionActivityUpdateRefCnt"
#define kAssertionCopyActivityUpdateOverflowKey "assertionActivityUpdateOverflow"
#define kAssertionCopyActivityUpdateDataKey     "assertionActivityUpdateData"
#define kAssertionCopyActivityUpdateProcessListKey "assertionActivityUpdateProcessList"
#define kAssertionCopyActiveAsyncAssertionsKey      "assertionActiveAsyncByProcess"

#define kPSAdapterDetails           "adapterDetails"

#define kInactivityWindowKey        "inactivityWindow"
#define kInactivityWindowStart      "inactivityWindowStart"
#define kInactivityWindowDuration   "inactivityWindowDuration"
#define kStandbyAccelerationDelay   "standbyAccelerationDelay"

#endif // _powermanagement_mig_h_
