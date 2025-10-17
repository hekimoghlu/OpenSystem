/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

//
//  SOSCircleRings.h
//  sec
//
//  Created by Richard Murphy on 12/4/14.
//
//

#ifndef sec_SOSCircleRings_h
#define sec_SOSCircleRings_h

__BEGIN_DECLS

/* return the ring recorded within the circle */
CFMutableSetRef SOSCircleGetRing(SOSCircleRef circle, CFStringRef ring);

/* return a set of peers referenced by a ring within the circle */
CFMutableSetRef SOSCircleRingCopyPeers(SOSCircleRef circle, CFStringRef ring, CFAllocatorRef allocator);

/* return the number of peers represented within a ring */
int SOSCircleRingCountPeers(SOSCircleRef circle, CFStringRef ring);

/* For each Peer in the circle, evaluate the ones purported to be allowed within a ring and sign them in to the ring */
bool SOSCircleRingAddPeers(SOSCircleRef oldCircle, SOSCircleRef newCircle, CFStringRef ring);

__END_DECLS

#endif
