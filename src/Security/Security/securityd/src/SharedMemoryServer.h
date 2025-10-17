/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
#ifndef __SHARED_MEMORY_SERVER__
#define __SHARED_MEMORY_SERVER__

#include <stdlib.h>
#include <string>
#include "SharedMemoryCommon.h"

class SharedMemoryServer
{
protected:
	std::string mSegmentName, mFileName;
	size_t mSegmentSize;
    uid_t mUID;

    u_int8_t* mSegment;
	u_int8_t* mDataArea;
	u_int8_t* mDataPtr;
	u_int8_t* mDataMax;

    int mBackingFile;
	
	void WriteOffset (SegmentOffsetType offset);
	void WriteData (const void* data, SegmentOffsetType length);


public:
	SharedMemoryServer (const char* segmentName, SegmentOffsetType segmentSize, uid_t uid = 0, gid_t gid = 0);
	virtual ~SharedMemoryServer ();
	
	void WriteMessage (SegmentOffsetType domain, SegmentOffsetType event, const void *message, SegmentOffsetType messageLength);
	
	const char* GetSegmentName ();
	size_t GetSegmentSize ();
	
	void SetProducerOffset (SegmentOffsetType producerOffset);
};



#endif
