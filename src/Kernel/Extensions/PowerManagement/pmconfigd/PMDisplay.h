/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#ifndef PMDisplay_h
#define PMDisplay_h
#include <SkyLight/SLSLegacy.h>
#include <CoreGraphics/CGError.h>
#include <CoreGraphics/CGSDisplay.h>
#include <IOKit/pwr_mgt/powermanagement_mig.h>
#include <sys/queue.h>
#include "PrivateLib.h"

// delay in seconds for clamshell reevalaute after
// wake
#define kClamshellEvaluateDelay 5

__private_extern__ void dimDisplay(void);
__private_extern__ void unblankDisplay(void);
__private_extern__ void blankDisplay(void);
__private_extern__ bool canSustainFullWake(void);
__private_extern__ void updateDesktopMode(xpc_object_t connection, xpc_object_t msg);
#if (TARGET_OS_OSX && TARGET_CPU_ARM64)
__private_extern__ void skylightCheckIn(xpc_object_t connection, xpc_object_t msg);
#endif
__private_extern__ bool skylightDisplayOn(void);
__private_extern__ bool isDesktopMode(void);
__private_extern__ void evaluateClamshellSleepState(void);
__private_extern__ void updateClamshellState(void *message);
__private_extern__ uint64_t inFlightDimRequest(void);
__private_extern__ void resetDisplayState(void);

void requestDisplayState(uint64_t state, int timeout);
void requestClamshellState(SLSClamshellState state);
void displayStateDidChange(uint64_t state);
void getClamshellDisplay(void);
#if (TARGET_OS_OSX && TARGET_CPU_ARM64)
void handleSkylightCheckIn(void);
#endif
void handleDesktopMode(void);
uint32_t rootDomainClamshellState(void);
void initializeClamshellState(void);

#if XCTEST
void xctSetDesktopMode(bool);
void xctSetClamshellState(uint32_t state);
uint32_t xctGetClamshellState(void);
#endif

#endif /* PMDisplay_h */
