/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#ifndef _PMConnection_h_
#define _PMConnection_h_


#define LOG_SLEEPSERVICES 1
/*
 * Struct for gSleepService 
 */
typedef struct {
    int                         notifyToken;
    CFStringRef                 uuid;
    long                        capTime;
} SleepServiceStruct;

// Bits for gPowerState
#define kSleepState                     0x01
#define kDarkWakeState                  0x02
#define kDarkWakeForBTState             0x04
#define kDarkWakeForSSState             0x08
#define kDarkWakeForMntceState          0x10
#define kDarkWakeForServerState         0x20
#define kFullWakeState                  0x40
#define kNotificationDisplayWakeState   0x80
#define kPowerStateMask                 0xff


#define IS_EMERGENCY_SLEEP(reason) \
                ((CFEqual((reason), CFSTR(kIOPMLowPowerSleepKey)) ||  \
                CFEqual((reason), CFSTR(kIOPMThermalEmergencySleepKey)) || \
                ((getSystemThermalState() != kIOPMThermalLevelNormal) && \
                 (getSystemThermalState() != kIOPMThermalLevelUnknown)))?true:false)

__private_extern__ void PMConnection_prime(void);

// PMAssertions.c calls into this when a PreventSystemSleep assertion is taken
__private_extern__ IOReturn _unclamp_silent_running(bool sendNewCapBits);
__private_extern__ IOReturn _clamp_silent_running(void);
__private_extern__ bool isInSilentRunningMode(void);

__private_extern__ IOReturn setRestrictedPerfMode(bool restrictPerf);
__private_extern__ bool isInPerfRestrictedMode(void);
__private_extern__ void evaluatePerfMode(void);

__private_extern__ bool _can_revert_sleep(void);
__private_extern__ void _set_sleep_revert(bool state);

__private_extern__ bool _woke_up_after_lastcall(void);

__private_extern__ io_connect_t getRootDomainConnect(void);
__private_extern__ bool isA_BTMtnceWake(void);
__private_extern__ bool isA_SleepSrvcWake(void);
__private_extern__ void set_SleepSrvcWake(void);
__private_extern__ bool isA_FullWake(void);
__private_extern__ void cancelPowerNapStates(void);

__private_extern__ bool isA_SleepState(void);
__private_extern__ bool isA_DarkWakeState(void);
__private_extern__ bool isA_NotificationDisplayWake(void);
__private_extern__ void set_NotificationDisplayWake(void);
__private_extern__ void cancel_NotificationDisplayWake(void);
__private_extern__ bool isCapabilityChangeDone(void);

__private_extern__ void InternalEvalConnections(void);
__private_extern__ kern_return_t getPlatformSleepType(uint32_t *sleepType, uint32_t *standbyTimer);
__private_extern__ void setDwlInterval(uint32_t newInterval);
__private_extern__ int getBTWakeInterval(void);
__private_extern__ uint64_t getCurrentWakeTime(void);
__private_extern__ void updateWakeTime(void);
__private_extern__ void updateCurrentWakeStart(uint64_t timestamp);
__private_extern__ void updateCurrentWakeEnd(uint64_t timestamp);
__private_extern__ void getScheduledWake(xpc_object_t remote, xpc_object_t msg);
__private_extern__ bool isEmergencySleep(void);
__private_extern__ int getCurrentSleepServiceCapTimeout(void);
/** Sets whether processes should get modified vm behavior for darkwake. */
__private_extern__ void setVMDarkwakeMode(bool darkwakeMode);
__private_extern__ void cancelDarkWakeCapabilitiesTimer(void);

#ifdef XCTEST
__private_extern__ void xctSetPowerState(uint32_t powerState);
#endif
#endif

