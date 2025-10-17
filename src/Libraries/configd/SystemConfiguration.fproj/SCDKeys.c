/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
/*
 * Modification History
 *
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * December 11, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include <SystemConfiguration/SystemConfiguration.h>

#include <stdarg.h>

/*
 * SCDynamicStoreKeyCreate*
 * - convenience routines that create a CFString key for an item in the store
 */

/*
 * Function: SCDynamicStoreKeyCreate
 * Purpose:
 *    Creates a store key using the given format.
 */
CFStringRef
SCDynamicStoreKeyCreate(CFAllocatorRef	allocator,
			CFStringRef	fmt,
			...)
{
	va_list		args;
	CFStringRef	result;

	va_start(args, fmt);
	result = CFStringCreateWithFormatAndArguments(allocator,
						      NULL,
						      fmt,
						      args);
	va_end(args);

	return result;
}

CFStringRef
SCDynamicStoreKeyCreateNetworkGlobalEntity(CFAllocatorRef	allocator,
					   CFStringRef		domain,
					   CFStringRef		entity)
{
	return (CFStringCreateWithFormat(allocator,
					 NULL,
					 CFSTR("%@/%@/%@/%@"),
					 domain,
					 kSCCompNetwork,
					 kSCCompGlobal,
					 entity));
}

CFStringRef
SCDynamicStoreKeyCreateNetworkInterface(CFAllocatorRef	allocator,
					CFStringRef	domain)
{
	return (CFStringCreateWithFormat(allocator,
					 NULL,
					 CFSTR("%@/%@/%@"),
					 domain,
					 kSCCompNetwork,
					 kSCCompInterface));
}

CFStringRef
SCDynamicStoreKeyCreateNetworkInterfaceEntity(CFAllocatorRef	allocator,
					      CFStringRef	domain,
					      CFStringRef	ifname,
					      CFStringRef	entity)
{
	if (entity == NULL) {
		return (CFStringCreateWithFormat(allocator,
						 NULL,
						 CFSTR("%@/%@/%@/%@"),
						 domain,
						 kSCCompNetwork,
						 kSCCompInterface,
						 ifname));
	} else {
		return (CFStringCreateWithFormat(allocator,
						 NULL,
						 CFSTR("%@/%@/%@/%@/%@"),
						 domain,
						 kSCCompNetwork,
						 kSCCompInterface,
						 ifname,
						 entity));
	}
}

CFStringRef
SCDynamicStoreKeyCreateNetworkServiceEntity(CFAllocatorRef	allocator,
					    CFStringRef		domain,
					    CFStringRef 	serviceID,
					    CFStringRef		entity)
{
	if (entity == NULL) {
		return (CFStringCreateWithFormat(allocator,
						 NULL,
						 CFSTR("%@/%@/%@/%@"),
						 domain,
						 kSCCompNetwork,
						 kSCCompService,
						 serviceID));
	} else {
		return (CFStringCreateWithFormat(allocator,
						 NULL,
						 CFSTR("%@/%@/%@/%@/%@"),
						 domain,
						 kSCCompNetwork,
						 kSCCompService,
						 serviceID,
						 entity));
	}
}
