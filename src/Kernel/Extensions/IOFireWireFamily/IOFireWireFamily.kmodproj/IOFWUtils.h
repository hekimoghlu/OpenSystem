/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#import <IOKit/IOTypes.h>

////////////////////////////////////////////////////////////////////////////////
//
// Useful FireWire utility functions.
//

#ifdef __cplusplus
extern "C" {
#endif

UInt16 FWComputeCRC16(const UInt32 *pQuads, UInt32 numQuads);
UInt16 FWUpdateCRC16(UInt16 crc16, UInt32 quad);

UInt32 AddFWCycleTimeToFWCycleTime( UInt32 cycleTime1, UInt32 cycleTime2 );
UInt32 SubtractFWCycleTimeFromFWCycleTime( UInt32 cycleTime1, UInt32 cycleTime2);

void IOFWGetAbsoluteTime( AbsoluteTime * result );
	
#ifdef __cplusplus
}
#endif

