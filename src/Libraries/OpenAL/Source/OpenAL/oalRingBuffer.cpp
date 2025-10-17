/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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
#include "oalRingBuffer.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#pragma mark ***** OALRingBuffer *****
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OALRingBuffer::OALRingBuffer() : mBuffer(NULL), mCapacityFrames(0), mCapacityBytes(0)
{
}

OALRingBuffer::OALRingBuffer(UInt32 bytesPerFrame, UInt32 capacityFrames) :
mBuffer(NULL)
{
    Allocate(bytesPerFrame, capacityFrames);
}

OALRingBuffer::~OALRingBuffer()
{
    Deallocate();
}

void	OALRingBuffer::Allocate(UInt32 bytesPerFrame, UInt32 capacityFrames)
{
	Deallocate();
	mBytesPerFrame = bytesPerFrame;
	mCapacityFrames = capacityFrames;
	mCapacityBytes = bytesPerFrame * capacityFrames;
	mBuffer = (Byte *)malloc(mCapacityBytes);
	Clear();
}

void	OALRingBuffer::Deallocate()
{
	if (mBuffer) {
		free(mBuffer);
		mBuffer = NULL;
	}
	mCapacityBytes = 0;
	mCapacityFrames = 0;
	Clear();
}

void	OALRingBuffer::Clear()
{
	if (mBuffer)
		memset(mBuffer, 0, mCapacityBytes);
	mStartOffset = 0;
	mStartFrame = 0;
	mEndFrame = 0;
}

bool	OALRingBuffer::Store(const Byte *data, UInt32 nFrames, SInt64 startFrame)
{
	if (nFrames > mCapacityFrames) return false;
    
	// reading and writing could well be in separate threads
    
	SInt64 endFrame = startFrame + nFrames;
	if (startFrame >= mEndFrame + mCapacityFrames)
		// writing more than one buffer ahead -- fine but that means that everything we have is now too far in the past
		Clear();
    
	if (mStartFrame == 0) {
		// empty buffer
		mStartOffset = 0;
		mStartFrame = startFrame;
		mEndFrame = endFrame;
		memcpy(mBuffer, data, nFrames * mBytesPerFrame);
	} else {
		UInt32 offset0, offset1, nBytes;
		if (endFrame > mEndFrame) {
			// advancing (as will be usual with sequential stores)
			
			if (startFrame > mEndFrame) {
				// we are skipping some samples, so zero the range we are skipping
				offset0 = FrameOffset(mEndFrame);
				offset1 = FrameOffset(startFrame);
				if (offset0 < offset1)
					memset(mBuffer + offset0, 0, offset1 - offset0);
				else {
					nBytes = mCapacityBytes - offset0;
					memset(mBuffer + offset0, 0, nBytes);
					memset(mBuffer, 0, offset1);
				}
			}
			mEndFrame = endFrame;
            
			// except for the case of not having wrapped yet, we will normally
			// have to advance the start
			SInt64 newStart = mEndFrame - mCapacityFrames;
			if (newStart > mStartFrame) {
				mStartOffset = (mStartOffset + (newStart - mStartFrame) * mBytesPerFrame) % mCapacityBytes;
				mStartFrame = newStart;
			}
		}
		// now everything is lined up and we can just write the new data
		offset0 = FrameOffset(startFrame);
		offset1 = FrameOffset(endFrame);
		if (offset0 < offset1)
			memcpy(mBuffer + offset0, data, offset1 - offset0);
		else {
			nBytes = mCapacityBytes - offset0;
			memcpy(mBuffer + offset0, data, nBytes);
			memcpy(mBuffer, data + nBytes, offset1);
		}
	}
	return true;
}

OSStatus	OALRingBuffer::Fetch(Byte *data, UInt32 nFrames, SInt64 startFrame)
{
	SInt64 endFrame = startFrame + nFrames;
	if (startFrame < mStartFrame || endFrame > mEndFrame) {
		return -1;
	}
	
	UInt32 offset0 = FrameOffset(startFrame);
	UInt32 offset1 = FrameOffset(endFrame);
	
	if (offset0 < offset1)
		memcpy(data, mBuffer + offset0, offset1 - offset0);
	else {
		UInt32 nBytes = mCapacityBytes - offset0;
		memcpy(data, mBuffer + offset0, nBytes);
		memcpy(data + nBytes, mBuffer, offset1);
	}
	return noErr;
}

Byte *	OALRingBuffer::GetFramePtr(SInt64 frameNumber, UInt32 &outNFrames)
{
	if (frameNumber < mStartFrame || frameNumber >= mEndFrame) {
		outNFrames = 0;
		return NULL;
	}
	UInt32 offset0 = FrameOffset(frameNumber);
	UInt32 offset1 = FrameOffset(mEndFrame);
	if (offset0 < offset1) {
		outNFrames = static_cast<UInt32>(mEndFrame - frameNumber);
	} else {
		outNFrames = (mCapacityBytes - offset0) / mBytesPerFrame;
	}
	return mBuffer + offset0;
}
