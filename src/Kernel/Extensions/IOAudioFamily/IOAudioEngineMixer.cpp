/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#include "IOAudioDebug.h"
#include "IOAudioEngine.h"
#include "IOAudioStream.h"
#include "IOAudioTypes.h"

#include <vecLib/vecLib.h>																							// <rdar://14058728>

IOReturn IOAudioEngine::mixOutputSamples(const void *sourceBuf, void *mixBuf, UInt32 firstSampleFrame, UInt32 numSampleFrames, const IOAudioStreamFormat *streamFormat, IOAudioStream *audioStream)
{
	IOReturn result = kIOReturnBadArgument;
	
    if (sourceBuf && mixBuf) 
	{
		const float * floatSource1Buf;
		const float * floatSource2Buf;
		float * floatMixBuf;
		
        UInt32 numSamplesLeft = numSampleFrames * streamFormat->fNumChannels;
		
		__darwin_size_t numSamps = numSamplesLeft;
 		
		floatMixBuf = &(((float *)mixBuf)[firstSampleFrame * streamFormat->fNumChannels]);
		floatSource2Buf = floatMixBuf;
		floatSource1Buf = (const float *)sourceBuf;
		
		__darwin_ptrdiff_t strideOne=1, strideTwo=1, resultStride=1;
		
		vDSP_vadd(floatSource1Buf, strideOne, floatSource2Buf, strideTwo, floatMixBuf, resultStride, numSamps);
		
        result = kIOReturnSuccess;
    }
    
    return result;
}
