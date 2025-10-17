/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#ifndef _SCD_H
#define _SCD_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <SystemConfiguration/SCDynamicStore.h>


typedef struct {
	int			_sc_error;		// SCError
	CFMutableDictionaryRef	_sc_interface_cache;	// SCNetworkInterface (cache)
	SCDynamicStoreRef	_sc_store;		// SCDynamicStore (null session)
} __SCThreadSpecificData, *__SCThreadSpecificDataRef;


__BEGIN_DECLS


#pragma mark -
#pragma mark [p]thread specific data


__SCThreadSpecificDataRef
__SCGetThreadSpecificData		(void);


#pragma mark -
#pragma mark ScheduleWithRunLoop/UnscheduleFromRunLoop


/*
 * object / CFRunLoop  management
 */
void
_SC_signalRunLoop			(CFTypeRef		obj,
					 CFRunLoopSourceRef     rls,
					 CFArrayRef		rlList);

Boolean
_SC_isScheduled				(CFTypeRef		obj,
					 CFRunLoopRef		runLoop,
					 CFStringRef		runLoopMode,
					 CFMutableArrayRef      rlList);

void
_SC_schedule				(CFTypeRef		obj,
					 CFRunLoopRef		runLoop,
					 CFStringRef		runLoopMode,
					 CFMutableArrayRef      rlList);

Boolean
_SC_unschedule				(CFTypeRef		obj,
					 CFRunLoopRef		runLoop,
					 CFStringRef		runLoopMode,
					 CFMutableArrayRef      rlList,
					 Boolean		all);


#pragma mark -
#pragma mark Misc


char *
_SC_cfstring_to_cstring_ext		(CFStringRef		cfstr,
					 char			*buf,
					 CFIndex		bufLen,
					 CFStringEncoding	encoding,
					 UInt8			lossByte,
					 CFIndex		*usedBufLen);

__END_DECLS

#endif	/* _SCD_H */
