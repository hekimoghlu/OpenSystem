/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#include <libkern/OSAtomic.h>

// Local includes
#include "oalImp.h"
#include "oalCaptureMixer.h"

// Public Utility includes
#include  "CAXException.h"

const UInt32 kMaxFramesPerSlice = 4096;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#pragma mark ***** OALCaptureMixers *****
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OALCaptureMixer::OALCaptureMixer (AudioUnit inMixerUnit, Float64 inSampleRate, UInt32 inOALFormat, UInt32 inBufferSize)
: mMixerUnit(inMixerUnit),
mRingBuffer(NULL),
mRequestedRingFrames(inBufferSize),
mStoreSampleTime(0),
mFetchSampleTime(0),
mCaptureOn(false),
mAudioConverter(0),
mConvertedDataABL(NULL)
{
	try {
        // translate the sample rate and data format parameters into an ASBD - this method throws
		FillInASBD(mRequestedFormat, inOALFormat, inSampleRate);
        
		// inBufferSize must be at least as big as one packet of the requested output format
		if (inBufferSize < mRequestedFormat.mBytesPerPacket)
            throw ((OSStatus) AL_INVALID_VALUE);
        
        // get the output format of the 3DMixer
        CAStreamBasicDescription mixerOutFormat;
        UInt32 propSize = sizeof(mixerOutFormat);
        XThrowIfErr(AudioUnitGetProperty(mMixerUnit, kAudioUnitProperty_StreamFormat, 0, 0, &mixerOutFormat, &propSize));
                
        mRingBuffer =  new OALRingBuffer();
		mRingBuffer->Allocate(mRequestedFormat.mBytesPerFrame, kMaxFramesPerSlice);
        
        //create a new converter to convert the mixer output data to the requested format
        XThrowIfErr(AudioConverterNew(&mixerOutFormat, &mRequestedFormat, &mAudioConverter));
        
        mConvertedDataABL = CABufferList::New("converted data", mRequestedFormat);
        mConvertedDataABL->AllocateBuffers(mRequestedFormat.mBytesPerFrame * kMaxFramesPerSlice);
        
	}
	catch (OSStatus	result) {
        if (mRingBuffer) {
            delete (mRingBuffer);
            mRingBuffer = NULL;
        }
        throw result;
	}
	catch (...) {
        if (mRingBuffer) {
            delete (mRingBuffer);
            mRingBuffer = NULL;
        }
        throw static_cast<OSStatus>(-1);
	}
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OALCaptureMixer::~OALCaptureMixer()
{
    if (mCaptureOn)
        StopCapture();
        
    if (mRingBuffer) {
        delete (mRingBuffer);
        mRingBuffer = NULL;
    }
    if (mConvertedDataABL) {
        delete (mConvertedDataABL);
        mConvertedDataABL = NULL;
    }
    if (mAudioConverter) {
        AudioConverterDispose(mAudioConverter);
        mAudioConverter = NULL;
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void OALCaptureMixer::StartCapture()
{
    if (!mCaptureOn)
    {
        SetCaptureFlag();
        AudioUnitAddRenderNotify(mMixerUnit, RenderCallback, this);
        mRingBuffer->Clear();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void OALCaptureMixer::StopCapture()
{
    if (mCaptureOn)
    {
        ClearCaptureFlag();
        AudioUnitRemoveRenderNotify(mMixerUnit, RenderCallback, this);
        mRingBuffer->Clear();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OSStatus OALCaptureMixer::RenderCallback(      void *							inRefCon,
                                               AudioUnitRenderActionFlags *	ioActionFlags,
                                               const AudioTimeStamp *			inTimeStamp,
                                               UInt32							inBusNumber,
                                               UInt32							inNumberFrames,
                                               AudioBufferList *				ioData)
{
    OALCaptureMixer* THIS = (OALCaptureMixer*) inRefCon;
    
    if (!THIS->mCaptureOn)
        return noErr;
    
    try {
        if (*ioActionFlags & kAudioUnitRenderAction_PostRender)
        {
            static int TEMP_kAudioUnitRenderAction_PostRenderError	= (1 << 8);
            if (inBusNumber == 0 && !(*ioActionFlags & TEMP_kAudioUnitRenderAction_PostRenderError))
            {
                // convert the data from the mixer to the requested format
                UInt32 packetCount = inNumberFrames;
                AudioBufferList* abl = &THIS->mConvertedDataABL->GetModifiableBufferList();
                XThrowIfErr(AudioConverterFillComplexBuffer(THIS->mAudioConverter, ConverterProc, ioData, &packetCount, abl, NULL));
                
                // store the converted data in the ring buffer
                if(THIS->mRingBuffer->Store((const Byte*)abl->mBuffers[0].mData, inNumberFrames, THIS->mStoreSampleTime))
                    THIS->mStoreSampleTime += inNumberFrames;
            }
        }
    }
    
    catch (OSStatus	result) {
        return result;
	}
	catch (...) {
        return -1;
	}
    
    return noErr;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OSStatus OALCaptureMixer::ConverterProc(       AudioConverterRef               /*inAudioConverter*/,
                                               UInt32*                           ioNumberDataPackets,
                                               AudioBufferList*                  ioData,
                                               AudioStreamPacketDescription**    /*outDataPacketDescription*/,
                                               void*                             inUserData)
{
    AudioBufferList* inputData = (AudioBufferList*) inUserData;
    for (int i=0; i<inputData->mNumberBuffers; ++i)
    {
        //tell the converter where the data is and how much there is
        ioData->mBuffers[i].mData = inputData->mBuffers[i].mData;
        ioData->mBuffers[i].mDataByteSize = inputData->mBuffers[i].mDataByteSize;
    }
        
    return noErr;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OSStatus	OALCaptureMixer::GetFrames(UInt32 inFrameCount, UInt8*	inBuffer)
{
	OSStatus	result = noErr;
	
	if (!mCaptureOn)
    {
		throw ((OSStatus) AL_INVALID_OPERATION);  // error condition, device not currently capturing
    }
    
	if (inFrameCount > AvailableFrames())
	{
		return -1; // error condition, there aren't enough valid frames to satisfy request
	}
	
	result = mRingBuffer->Fetch((Byte*)inBuffer, inFrameCount, mFetchSampleTime);
	
	if (result)
	{
		return result;
	}
	
	if (result == noErr)
		mFetchSampleTime += inFrameCount;
	   
	return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
UInt32	OALCaptureMixer::AvailableFrames()
{
	SInt64	start, end;
	mRingBuffer->GetTimeBounds(start, end);
    
	if (mFetchSampleTime < start)
		mFetchSampleTime = start;	// move up our fetch starting point, we have fallen too far behind
	
	UInt32	availableFrames = static_cast<UInt32>(end - mFetchSampleTime);
    
	if (availableFrames > mRequestedRingFrames)
		availableFrames = mRequestedRingFrames;
    
	return availableFrames;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void	OALCaptureMixer::SetCaptureFlag()
{
    int32_t one = 1;    int32_t zero = 0;
    OSAtomicCompareAndSwap32Barrier(zero, one, &mCaptureOn);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void	OALCaptureMixer::ClearCaptureFlag()
{
    int32_t one = 1;    int32_t zero = 0;
    OSAtomicCompareAndSwap32Barrier(one, zero, &mCaptureOn);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




