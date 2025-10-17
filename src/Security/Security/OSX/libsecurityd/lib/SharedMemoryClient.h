/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
#ifndef __SHAREDMEMORYCLIENT__
#define __SHAREDMEMORYCLIENT__

#include <string>
#include <stdlib.h>
#include <securityd_client/SharedMemoryCommon.h>
#include <security_utilities/threading.h>

namespace Security
{

    
enum UnavailableReason {kURNone, kURMessageDropped, kURMessagePending, kURNoMessage, kURBufferCorrupt};

class SharedMemoryClient
{
protected:
	std::string mSegmentName;
	size_t mSegmentSize;
	Mutex mMutex;
    uid_t mUID;

	u_int8_t* mSegment;
	u_int8_t* mDataArea;
	u_int8_t* mDataPtr;
	u_int8_t* mDataMax;
	
	SegmentOffsetType GetProducerCount ();

	void ReadData (void* buffer, SegmentOffsetType bytesToRead);
	SegmentOffsetType ReadOffset ();
	
public:
	SharedMemoryClient (const char* segmentName, SegmentOffsetType segmentSize, uid_t uid = 0);
	virtual ~SharedMemoryClient ();
	
	bool ReadMessage (void* message, SegmentOffsetType &length, UnavailableReason &ur);
	
    const char* GetSegmentName() { return mSegmentName.c_str (); }
    size_t GetSegmentSize() { return mSegmentSize; }

    uid_t getUID() const { return mUID; }

    bool uninitialized() { return (mSegment == NULL || mSegment == MAP_FAILED); }
};

};  /* namespace */


#endif

