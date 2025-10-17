/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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
/*!
    @header IOPMKeys.h
    
    IOPMKeys.h defines C strings for use accessing power management data.
    Note that all of these C strings must be converted to CFStrings before use. You can wrap
    them with the CFSTR() macro, or create a CFStringRef (that you must later CFRelease()) using CFStringCreateWithCString()
 */
 
#ifndef _IOPMKEYS_H_
#define _IOPMKEYS_H_
 
/*
 * Types of power event
 * These are potential arguments to IOPMSchedulePowerEvent().
 * These are all potential values of the kIOPMPowerEventTypeKey in the CFDictionaries
 * returned by IOPMCopyScheduledPowerEvents().
 */
/*!
    @define kIOPMAutoWake 
    @abstract Value for scheduled wake from sleep.
*/
#define kIOPMAutoWake                                   "wake"

/*!
    @define kIOPMAutoPowerOn 
    @abstract Value for scheduled power on from off state.
*/
#define kIOPMAutoPowerOn                                "poweron"

/*!
    @define kIOPMAutoWakeOrPowerOn 
    @abstract Value for scheduled wake from sleep, or power on. The system will either wake OR
        power on, whichever is necessary.
*/

#define kIOPMAutoWakeOrPowerOn                          "wakepoweron"
/*!
    @define kIOPMAutoSleep
    @abstract Value for scheduled sleep.
*/

#define kIOPMAutoSleep                                   "sleep"
/*!
    @define kIOPMAutoShutdown 
    @abstract Value for scheduled shutdown.
*/

#define kIOPMAutoShutdown                               "shutdown"
/*!
    @define kIOPMAutoRestart 
    @abstract Value for scheduled restart.
*/

#define kIOPMAutoRestart                                "restart"

/*
 * Keys for evaluating the CFDictionaries returned by IOPMCopyScheduledPowerEvents()
 */
/*!
    @define kIOPMPowerEventTimeKey 
    @abstract Key for the time of the scheduled power event. Value is a CFDateRef.
*/
#define kIOPMPowerEventTimeKey                          "time"

/*!
    @define kIOPMPowerEventAppNameKey 
    @abstract Key for the CFBundleIdentifier of the app that scheduled the power event. Value is a CFStringRef.
*/
#define kIOPMPowerEventAppNameKey                       "scheduledby"

/*!
 @define kIOPMPowerEventAppPIDKey
 @abstract Key for the PID the App that scheduled the power event. Value is a CFNumber integer.
 */
#define kIOPMPowerEventAppPIDKey                       "appPID"

/*!
    @define kIOPMPowerEventTypeKey 
    @abstract Key for the type of power event. Value is a CFStringRef, with the c-string value of one of the "kIOPMAuto" strings.
*/
#define kIOPMPowerEventTypeKey                          "eventtype"

#endif
