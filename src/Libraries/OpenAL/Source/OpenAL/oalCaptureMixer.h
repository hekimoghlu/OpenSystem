/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#ifndef __OpenAL__oalCaptureMixer__
#define __OpenAL__oalCaptureMixer__

//local includes
#include "oalRingBuffer.h"

// System includes
#include <AudioToolbox/AudioToolbox.h>

// Public Utility includes
#include  "CAStreamBasicDescription.h"
#include  "CABufferList.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// OALCaptureMixer
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#pragma mark _____OALCaptureMixer_____
class OALCaptureMixer
{
public:
    OALCaptureMixer(AudioUnit inMixer, Float64 inSampleRate, UInt32 inOALFormat, UInt32 inBufferSize);
    ~OALCaptureMixer();
    
    void						StartCapture();
	void						StopCapture();
    OSStatus					GetFrames(UInt32 inFrameCount, UInt8*	inBuffer);
	UInt32						AvailableFrames();
    CAStreamBasicDescription	GetOutputFormat() { return mRequestedFormat; }
    
    static OSStatus RenderCallback( void *							inRefCon,
                                   AudioUnitRenderActionFlags *     ioActionFlags,
                                   const AudioTimeStamp *			inTimeStamp,
                                   UInt32							inBusNumber,
                                   UInt32							inNumberFrames,
                                   AudioBufferList *				ioData);
    
    static OSStatus ConverterProc(  AudioConverterRef               inAudioConverter,
                                  UInt32*                           ioNumberDataPackets,
                                  AudioBufferList*                  ioData,
                                  AudioStreamPacketDescription**    outDataPacketDescription,
                                  void*                             inUserData);
    
    void                        SetCaptureFlag();
    void                        ClearCaptureFlag();
    bool                        IsCapturing() { return static_cast<bool>(mCaptureOn); }
    
private:
    AudioUnit                   mMixerUnit;
    CAStreamBasicDescription    mRequestedFormat;
    OALRingBuffer*              mRingBuffer;
    UInt32                      mRequestedRingFrames;
    SInt64                      mStoreSampleTime;
    SInt64                      mFetchSampleTime;
    volatile int32_t            mCaptureOn;
    
    AudioConverterRef           mAudioConverter;
    CABufferList*               mConvertedDataABL;
};

#endif /* defined(__OpenAL__oalCaptureMixer__) */
