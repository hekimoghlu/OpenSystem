/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#ifndef PvDInfoContext_h
#define PvDInfoContext_h

#import <CoreFoundation/CoreFoundation.h>

CF_ASSUME_NONNULL_BEGIN

#define kPvDInfoClientRefetchSamePvDIDMinWaitSeconds 10

typedef struct {
	CFStringRef pvdid;
	CFArrayRef _Nullable ipv6_prefixes;
	const char * if_name;
	uint16_t sequence_nr;
	bool status_ok;
	CFDictionaryRef _Nullable additional_info;
	CFDateRef last_fetched_date;
	CFDateRef effective_expiration_date;
} PvDInfoContext;

void
PvDInfoContextFlush(PvDInfoContext * ret_context, bool persist_failure);

CFStringRef
PvDInfoContextGetPvDID(PvDInfoContext * ret_context);

void
PvDInfoContextSetPvDID(PvDInfoContext * ret_context, CFStringRef pvdid);

CFArrayRef
PvDInfoContextGetPrefixes(PvDInfoContext * ret_context);

void
PvDInfoContextSetPrefixes(PvDInfoContext * ret_context,
			  CFArrayRef _Nullable prefixes);

const char *
PvDInfoContextGetInterfaceName(PvDInfoContext * ret_context);

void
PvDInfoContextSetInterfaceName(PvDInfoContext * ret_context,
			       const char * if_name);

uint16_t
PvDInfoContextGetSequenceNumber(PvDInfoContext * ret_context);

void
PvDInfoContextSetSequenceNumber(PvDInfoContext * ret_context, uint16_t seqnr);

bool
PvDInfoContextIsOk(PvDInfoContext * ret_context);

void
PvDInfoContextSetIsOk(PvDInfoContext * ret_context, bool ok);

CFDictionaryRef
PvDInfoContextGetAdditionalInformation(PvDInfoContext * ret_context);

void
PvDInfoContextSetAdditionalInformation(PvDInfoContext * ret_context,
				       CFDictionaryRef _Nullable additional_info);

bool
PvDInfoContextCanRefetch(PvDInfoContext * ret_context);

bool
PvDInfoContextFetchAllowed(PvDInfoContext * ret_context);

void
PvDInfoContextSetLastFetchedDateToNow(PvDInfoContext * ret_context);

CFAbsoluteTime
PvDInfoContextGetEffectiveExpirationTime(PvDInfoContext * ret_context);

void
PvDInfoContextCalculateEffectiveExpiration(PvDInfoContext * ret_context);

CF_ASSUME_NONNULL_END

#endif /* PvDInfoContext_h */
