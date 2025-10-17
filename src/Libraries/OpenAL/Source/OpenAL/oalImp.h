/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#ifndef __OAL_IMP__
#define __OAL_IMP__

#include "al.h"
#include "alc.h"
#include "oalOSX.h"
#include <Carbon/Carbon.h>
#include <map>

typedef ALvoid			(*alSourceNotificationProc) (ALuint sid, ALuint	notificationID, ALvoid*	userData);

#ifdef __cplusplus
extern "C" {
#endif

// added for OSX Extension 
ALC_API ALvoid alcMacOSXRenderingQuality (ALint value) OPENAL_DEPRECATED;
ALC_API ALvoid alMacOSXRenderChannelCount (ALint value) OPENAL_DEPRECATED;
ALC_API ALvoid alcMacOSXMixerMaxiumumBusses (ALint value) OPENAL_DEPRECATED;
ALC_API ALvoid alcMacOSXMixerOutputRate(ALdouble value) OPENAL_DEPRECATED;

ALC_API ALint alcMacOSXGetRenderingQuality () OPENAL_DEPRECATED;
ALC_API ALint alMacOSXGetRenderChannelCount () OPENAL_DEPRECATED;
ALC_API ALint alcMacOSXGetMixerMaxiumumBusses () OPENAL_DEPRECATED;
ALC_API ALdouble alcMacOSXGetMixerOutputRate() OPENAL_DEPRECATED;

AL_API ALvoid AL_APIENTRY alSetInteger (ALenum pname, ALint value) OPENAL_DEPRECATED;
AL_API ALvoid AL_APIENTRY alSetDouble (ALenum pname, ALdouble value) OPENAL_DEPRECATED;

AL_API ALvoid	AL_APIENTRY	alBufferDataStatic (ALint bid, ALenum format, const ALvoid* data, ALsizei size, ALsizei freq) OPENAL_DEPRECATED;

// source notifications
AL_API ALenum alSourceAddNotification (ALuint sid, ALuint notificationID, alSourceNotificationProc notifyProc, ALvoid* userData) OPENAL_DEPRECATED;
AL_API ALvoid alSourceRemoveNotification (ALuint	sid, ALuint notificationID, alSourceNotificationProc notifyProc, ALvoid* userData) OPENAL_DEPRECATED;
    
// source spatialization
AL_API ALvoid alSourceRenderingQuality (ALuint sid, ALint value) OPENAL_DEPRECATED;
AL_API ALint  alSourceGetRenderingQuality (ALuint sid) OPENAL_DEPRECATED;

// added for ASA (Apple Environmental Audio) 

ALC_API ALenum  alcASAGetSource(ALuint property, ALuint source, ALvoid *data, ALuint* dataSize) OPENAL_DEPRECATED;
ALC_API ALenum  alcASASetSource(ALuint property, ALuint source, ALvoid *data, ALuint dataSize) OPENAL_DEPRECATED;
ALC_API ALenum  alcASAGetListener(ALuint property, ALvoid *data, ALuint* dataSize) OPENAL_DEPRECATED;
ALC_API ALenum  alcASASetListener(ALuint property, ALvoid *data, ALuint dataSize) OPENAL_DEPRECATED;
    
// 3DMixer output capturer
ALC_API ALvoid  alcOutputCapturerPrepare( ALCuint frequency, ALCenum format, ALCsizei buffersize ) OPENAL_DEPRECATED;
ALC_API ALvoid  alcOutputCapturerStart() OPENAL_DEPRECATED;
ALC_API ALvoid  alcOutputCapturerStop() OPENAL_DEPRECATED;
ALC_API ALint   alcOutputCapturerAvailableSamples() OPENAL_DEPRECATED;
ALC_API ALvoid  alcOutputCapturerSamples( ALCvoid *buffer, ALCsizei samples ) OPENAL_DEPRECATED;

// Used internally but no longer available via a header file. Some OpenAL applications may have been built with a header
// that defined these constants so keep defining them.

#define ALC_SPATIAL_RENDERING_QUALITY        0xF002
#define ALC_MIXER_OUTPUT_RATE		         0xF003
#define ALC_MIXER_MAXIMUM_BUSSES             0xF004
#define ALC_RENDER_CHANNEL_COUNT             0xF005

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define AL_FORMAT_MONO_FLOAT32               0x10010
#define AL_FORMAT_STEREO_FLOAT32             0x10011

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Continue exporting these deprectaed APIs to prevent runtime link errors

#ifdef TARGET_OS_MAC
   #if TARGET_OS_MAC
       #pragma export on
   #endif
#endif


AL_API ALvoid	AL_APIENTRY alHint( ALenum target, ALenum mode );

#ifdef TARGET_OS_MAC
   #if TARGET_OS_MAC
      #pragma export off
   #endif
#endif

#ifdef __cplusplus
}
#endif

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// development build flags
#define	USE_AU_TRACER 					0
#define LOG_GRAPH_AND_MIXER_CHANGES		0
#define GET_OVERLOAD_NOTIFICATIONS 		0
#define	LOG_IO 							0

#if LOG_IO
	#include  "AudioLogger.h"
#endif

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define AL_MAXBUFFERS 1024	
#define AL_MAXSOURCES 256 

#define kDefaultMaximumMixerBusCount    64
#define kDopplerDefault                 0	

enum {
		kRogerBeepType	= 'rogr',
		kDistortionType	= 'dist'
};

#define	THROW_RESULT		if(result != noErr) throw static_cast<OSStatus>(result);

enum {
		kUnknown3DMixerVersion	= 0,
		kUnsupported3DMixer		= 1,
		k3DMixerVersion_1_3		= 13,
		k3DMixerVersion_2_0,
		k3DMixerVersion_2_1,
		k3DMixerVersion_2_2,
		k3DMixerVersion_2_3
};

enum {
		kUnknownAUState	= -1,
		kAUIsNotPresent	= 0,
		kAUIsPresent	= 1
};		
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void	alSetError (ALenum errorCode);
UInt32	Get3DMixerVersion ();
ALCint  IsDistortionPresent();
ALCint  IsRogerBeepPresent();

#endif

