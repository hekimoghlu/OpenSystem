/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
#include <AudioToolbox/AudioToolbox.h>

#ifndef __OpenAL_Aspen__oalRingBuffer__
#define __OpenAL_Aspen__oalRingBuffer__

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#pragma mark _____OALRingBuffer_____
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// OALRingBuffer:
// This class implements an audio ring buffer. Multi-channel data can be either
// interleaved or deinterleaved.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum {
	kOALRingBufferError_WayBehind = -2, // both fetch times are earlier than buffer start time
	kOALRingBufferError_SlightlyBehind = -1, // fetch start time is earlier than buffer start time (fetch end time OK)
	kOALRingBufferError_OK = 0,
	kOALRingBufferError_SlightlyAhead = 1, // fetch end time is later than buffer end time (fetch start time OK)
	kOALRingBufferError_WayAhead = 2, // both fetch times are later than buffer end time
	kOALRingBufferError_TooMuch = 3, // fetch start time is earlier than buffer start time and fetch end time is later than buffer end time
	kOALRingBufferError_CPUOverload = 4 // the reader is unable to get enough CPU cycles to capture a consistent snapshot of the time bounds
};

typedef SInt32 OALRingBufferError;
typedef SInt64 SampleTime;

const UInt32 kGeneralRingTimeBoundsQueueSize = 32;
const UInt32 kGeneralRingTimeBoundsQueueMask = kGeneralRingTimeBoundsQueueSize - 1;

class OALRingBuffer {
public:
	OALRingBuffer();
	OALRingBuffer(UInt32 bytesPerFrame, UInt32 capacityFrames);
	~OALRingBuffer();
	
	void		Allocate(UInt32 bytesPerFrame, UInt32 capacityFrames);
	void		Deallocate();
	void		Clear();
	bool		Store(const Byte *data, UInt32 nFrames, SInt64 frameNumber);
	OSStatus	Fetch(Byte *data, UInt32 nFrames, SInt64 frameNumber);
	Byte*		GetFramePtr(SInt64 frameNumber, UInt32 &outNFrames);
	
	void		GetTimeBounds(SInt64 &start, SInt64&end) { start = mStartFrame; end = mEndFrame; }
	
protected:
	UInt32		FrameOffset(SInt64 frameNumber) { return (mStartOffset + UInt32(frameNumber - mStartFrame) * mBytesPerFrame) % mCapacityBytes; }
    
protected:
	Byte *		mBuffer;
	UInt32		mBytesPerFrame;
	UInt32		mCapacityFrames;
	UInt32		mCapacityBytes;
	UInt32		mStartOffset;
	SInt64		mStartFrame;
	SInt64		mEndFrame;
};

#endif  /* defined(__OpenAL_Aspen__oalRingBuffer__) */