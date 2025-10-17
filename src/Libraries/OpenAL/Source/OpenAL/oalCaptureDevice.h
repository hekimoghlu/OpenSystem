/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
#ifndef __OAL_CAPTURE_DEVICE__
#define __OAL_CAPTURE_DEVICE__

#include <CoreAudio/AudioHardware.h>
#include <AudioToolbox/AudioToolbox.h>
#include <AudioUnit/AudioUnit.h>
#include <map>
#include <libkern/OSAtomic.h>

#include "oalImp.h"
#include "oalRingBuffer.h"
#include "CAStreamBasicDescription.h"
#include "CABufferList.h"

#define LOG_CAPTUREDEVICE_VERBOSE         0

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// OALCaptureDevices
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#pragma mark _____OALCaptureDevice_____
class OALCaptureDevice
{
#pragma mark __________ Public_Class_Members
	public:

	OALCaptureDevice(const char* 	 inDeviceName, uintptr_t   inSelfToken, UInt32 inSampleRate, UInt32 inFormat, UInt32 inBufferSize);
	~OALCaptureDevice();
	
	void				StartCapture();
	void				StopCapture();

	OSStatus			GetFrames(UInt32 inFrameCount, UInt8*	inBuffer);
	UInt32				AvailableFrames();
	void				SetError(ALenum errorCode);
	ALenum				GetError();

	// we need to mark the capture device if it is being used to prevent deletion from another thread
	void				SetInUseFlag()		{ OSAtomicIncrement32Barrier(&mInUseFlag); }	
	void				ClearInUseFlag()	{ OSAtomicDecrement32Barrier(&mInUseFlag); }
	volatile int32_t	IsInUse()			{ return mInUseFlag; }

#pragma mark __________ Private_Class_Members

	private:
#if LOG_CAPTUREDEVICE_VERBOSE
		uintptr_t						mSelfToken;
#endif
		ALenum							mCurrentError;
		bool							mCaptureOn;
		SInt64							mStoreSampleTime;				// increment on each read in the input proc, and pass to the ring buffer class when writing, reset on each stop
		SInt64							mFetchSampleTime;				// increment on each read in the input proc, and pass to the ring buffer class when writing, reset on each stop
		AudioUnit						mInputUnit;
		CAStreamBasicDescription		mNativeFormat;
		CAStreamBasicDescription		mRequestedFormat;
		CAStreamBasicDescription		mOutputFormat;
		OALRingBuffer*					mRingBuffer;					// the ring buffer
		UInt8*							mBufferData;
		AudioConverterRef				mAudioConverter;
		Float64							mSampleRateRatio;
		UInt32							mRequestedRingFrames;			
		CABufferList*					mAudioInputPtrs;
		volatile int32_t				mInUseFlag;						// flag to indicate the device is currently being used by one or more threads

	void				InitializeAU (const char* 	inDeviceName);
	static OSStatus		InputProc(	void *						inRefCon,
									AudioUnitRenderActionFlags *ioActionFlags,
									const AudioTimeStamp *		inTimeStamp,
									UInt32 						inBusNumber,
									UInt32 						inNumberFrames,
									AudioBufferList *			ioData);
									
	static OSStatus		ACComplexInputDataProc	(AudioConverterRef				inAudioConverter,
												 UInt32							*ioNumberDataPackets,
												 AudioBufferList				*ioData,
												 AudioStreamPacketDescription	**outDataPacketDescription,
												 void*							inUserData);
										
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#pragma mark _____OALCaptureDeviceMap_____
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OALCaptureDeviceMap : std::multimap<uintptr_t, OALCaptureDevice*, std::less<uintptr_t> > {
public:
    
    void Add (const	uintptr_t	inDeviceToken, OALCaptureDevice **inDevice)  {
		iterator it = upper_bound(inDeviceToken);
		insert(it, value_type (inDeviceToken, *inDevice));
	}

    OALCaptureDevice* Get(uintptr_t	inDeviceToken) {
        iterator	it = find(inDeviceToken);
        if (it != end())
            return ((*it).second);
		return (NULL);
    }
    
    void Remove (const	uintptr_t	inDeviceToken) {
        iterator 	it = find(inDeviceToken);
        if (it != end())
            erase(it);
    }
	
    UInt32 Size () const { return size(); }
    bool Empty () const { return empty(); }
};


#endif
