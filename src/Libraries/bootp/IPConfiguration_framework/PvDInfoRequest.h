/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#ifndef PvDInfoRequest_h
#define PvDInfoRequest_h

#import <CoreFoundation/CoreFoundation.h>

CF_ASSUME_NONNULL_BEGIN

typedef struct __PvDInfoRequest * PvDInfoRequestRef;

typedef CF_ENUM(uint32_t, PvDInfoRequestState) {
	kPvDInfoRequestStateIdle = 0,
	kPvDInfoRequestStateScheduled = 1,
	kPvDInfoRequestStateObtained = 2,
	kPvDInfoRequestStateFailed = 3
};

/*
 * PvDInfoRequestRef is a CFRuntime object.
 * Must be released with CFRelease().
 */
PvDInfoRequestRef
PvDInfoRequestCreate(CFStringRef pvdid, CFArrayRef prefixes,
		     const char * ifname, uint64_t ms_delay);

void
PvDInfoRequestSetCompletionHandler(PvDInfoRequestRef request,
				   dispatch_block_t completion,
				   dispatch_queue_t queue);

void
PvDInfoRequestCancel(PvDInfoRequestRef request);

void
PvDInfoRequestResume(PvDInfoRequestRef request);

PvDInfoRequestState
PvDInfoRequestGetCompletionStatus(PvDInfoRequestRef request);

CFDictionaryRef
PvDInfoRequestCopyAdditionalInformation(PvDInfoRequestRef request);

CF_ASSUME_NONNULL_END

#endif /* PvDInfoRequest_h */
