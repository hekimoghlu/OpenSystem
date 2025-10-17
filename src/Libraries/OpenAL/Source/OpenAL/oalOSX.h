/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#ifndef __OAL_OSX__
#define __OAL_OSX__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <Carbon/Carbon.h>
#include <AudioToolbox/AudioToolbox.h>
#include "CAStreamBasicDescription.h"
#include "oalImp.h"
#include "MacOSX_OALExtensions.h"

#define maxLen 256


bool		IsFormatSupported(UInt32	inFormatID);
OSStatus	FillInASBD(CAStreamBasicDescription &inASBD, UInt32	inFormatID, UInt32 inSampleRate);
bool		IsValidRenderQuality (UInt32 inRenderQuality);
UInt32		GetDesiredRenderChannelsFor3DMixer(UInt32	inDeviceChannels);
void		GetDefaultDeviceName(ALCchar*		outDeviceName, bool	isInput);
uintptr_t	GetNewPtrToken (void);
ALuint		GetNewToken (void);
UInt32		CalculateNeededMixerBusses(const ALCint *attrList, UInt32 inDefaultBusCount);
const char* GetFormatString(UInt32 inToken);
const char* GetALAttributeString(UInt32 inToken);
const char* GetALCAttributeString(UInt32 inToken);
void		WaitOneRenderCycle();
void		ReconfigureContextsOfThisDevice(uintptr_t inDeviceToken);

#endif
