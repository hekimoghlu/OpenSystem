/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#ifndef _OSTHERMALNOTIFICATION_H_
#define _OSTHERMALNOTIFICATION_H_

#include <_bounds.h>
#include <sys/cdefs.h>
#include <Availability.h>
#include <TargetConditionals.h>

_LIBC_SINGLE_BY_DEFAULT()

/*
**  OSThermalNotification.h
**  
**  Notification mechanism to alert registered tasks when device thermal conditions
**  reach certain thresholds. Notifications are triggered in both directions
**  so clients can manage their memory usage more and less aggressively.
**
*/

__BEGIN_DECLS

/* Define pressure levels usable by OSThermalPressureLevel */
typedef enum {
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
	kOSThermalPressureLevelNominal = 0,
	kOSThermalPressureLevelModerate,
	kOSThermalPressureLevelHeavy,
	kOSThermalPressureLevelTrapping,
	kOSThermalPressureLevelSleeping
#else
	kOSThermalPressureLevelNominal = 0,
	kOSThermalPressureLevelLight = 10,
	kOSThermalPressureLevelModerate = 20,
	kOSThermalPressureLevelHeavy = 30,
	kOSThermalPressureLevelTrapping = 40,
	kOSThermalPressureLevelSleeping = 50
#endif
} OSThermalPressureLevel;

/*
 ** External notify(3) string for thermal pressure level notification
 */
__OSX_AVAILABLE_STARTING(__MAC_10_10, __IPHONE_7_0)
extern const char * const kOSThermalNotificationPressureLevelName;


#if defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && \
	__IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_2_0

typedef enum {
	OSThermalNotificationLevelAny      = -1,
	OSThermalNotificationLevelNormal   =  0,
} OSThermalNotificationLevel;

extern OSThermalNotificationLevel _OSThermalNotificationLevelForBehavior(int) __OSX_AVAILABLE_STARTING(__MAC_NA, __IPHONE_4_2);
extern void _OSThermalNotificationSetLevelForBehavior(int, int) __OSX_AVAILABLE_STARTING(__MAC_NA, __IPHONE_4_2);

enum {
	kOSThermalMitigationNone,
	kOSThermalMitigation70PercentTorch,
	kOSThermalMitigation70PercentBacklight,
	kOSThermalMitigation50PercentTorch,
	kOSThermalMitigation50PercentBacklight,
	kOSThermalMitigationDisableTorch,
	kOSThermalMitigation25PercentBacklight,
	kOSThermalMitigationDisableMapsHalo,
	kOSThermalMitigationAppTerminate,
	kOSThermalMitigationDeviceRestart,
	kOSThermalMitigationThermalTableReady,
	kOSThermalMitigationCount
};

#define OSThermalNotificationLevel70PercentTorch _OSThermalNotificationLevelForBehavior(kOSThermalMitigation70PercentTorch)
#define OSThermalNotificationLevel70PercentBacklight _OSThermalNotificationLevelForBehavior(kOSThermalMitigation70PercentBacklight)
#define OSThermalNotificationLevel50PercentTorch _OSThermalNotificationLevelForBehavior(kOSThermalMitigation50PercentTorch)
#define OSThermalNotificationLevel50PercentBacklight _OSThermalNotificationLevelForBehavior(kOSThermalMitigation50PercentBacklight)
#define OSThermalNotificationLevelDisableTorch _OSThermalNotificationLevelForBehavior(kOSThermalMitigationDisableTorch)
#define OSThermalNotificationLevel25PercentBacklight _OSThermalNotificationLevelForBehavior(kOSThermalMitigation25PercentBacklight)
#define OSThermalNotificationLevelDisableMapsHalo _OSThermalNotificationLevelForBehavior(kOSThermalMitigationDisableMapsHalo)
#define OSThermalNotificationLevelAppTerminate _OSThermalNotificationLevelForBehavior(kOSThermalMitigationAppTerminate)
#define OSThermalNotificationLevelDeviceRestart _OSThermalNotificationLevelForBehavior(kOSThermalMitigationDeviceRestart)

/* Backwards compatibility */
#define OSThermalNotificationLevelWarning OSThermalNotificationLevel70PercentBacklight
#define OSThermalNotificationLevelUrgent OSThermalNotificationLevelAppTerminate
#define OSThermalNotificationLevelCritical OSThermalNotificationLevelDeviceRestart

/*
** Simple polling interface to detect current thermal level
*/
__OSX_AVAILABLE_STARTING(__MAC_NA, __IPHONE_2_0)
extern OSThermalNotificationLevel OSThermalNotificationCurrentLevel(void);

/*
** External notify(3) string for manual notification setup
*/
__OSX_AVAILABLE_STARTING(__MAC_NA, __IPHONE_2_0)
extern const char * const kOSThermalNotificationName;

/*
** External notify(3) string for alerting user of a thermal condition
*/
__OSX_AVAILABLE_STARTING(__MAC_NA, __IPHONE_6_0)
extern const char * const kOSThermalNotificationAlert;

/*
** External notify(3) string for notifying system the options taken to resolve thermal condition
*/
__OSX_AVAILABLE_STARTING(__MAC_NA, __IPHONE_6_0)
extern const char * const kOSThermalNotificationDecision;

#endif // __IPHONE_OS_VERSION_MIN_REQUIRED

__END_DECLS

#endif /* _OSTHERMALNOTIFICATION_H_ */
