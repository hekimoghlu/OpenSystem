/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
 * October 29, 2001		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCPreferencesPathKey.h>
#include <SystemConfiguration/SCSchemaDefinitionsPrivate.h>

#include <stdarg.h>

CFStringRef
SCPreferencesPathKeyCreate(CFAllocatorRef	allocator,
			   CFStringRef		fmt,
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
SCPreferencesPathKeyCreateNetworkServices(CFAllocatorRef	allocator)
{
	/*
	 * create "/NetworkServices"
	 */
	return CFStringCreateWithFormat(allocator,
					NULL,
					CFSTR("/%@"),
					kSCPrefNetworkServices);
}


CFStringRef
SCPreferencesPathKeyCreateNetworkServiceEntity(CFAllocatorRef	allocator,
					       CFStringRef	service,
					       CFStringRef	entity)
{
	CFStringRef path;

	if (entity == NULL) {
		/*
		 * create "/NetworkServices/service-id"
		 */
		path = CFStringCreateWithFormat(allocator,
						NULL,
						CFSTR("/%@/%@"),
						kSCPrefNetworkServices,
						service);
	} else {
		/*
		 * create "/NetworkServices/service-id/entity"
		 */
		path = CFStringCreateWithFormat(allocator,
						NULL,
						CFSTR("/%@/%@/%@"),
						kSCPrefNetworkServices,
						service,
						entity);
	}

	return path;
}


CFStringRef
SCPreferencesPathKeyCreateSets(CFAllocatorRef	allocator)
{
	/*
	 * create "/Sets"
	 */
	return (CFStringCreateWithFormat(allocator,
					 NULL,
					 CFSTR("/%@"),
					 kSCPrefSets));
}


CFStringRef
SCPreferencesPathKeyCreateSet(CFAllocatorRef	allocator,
			      CFStringRef       set)
{
	/*
	 * create "/Sets/set-id"
	 */
	return (CFStringCreateWithFormat(allocator,
					 NULL,
					 CFSTR("/%@/%@"),
					 kSCPrefSets,
					 set));
}


CFStringRef
SCPreferencesPathKeyCreateSetNetworkGlobalEntity(CFAllocatorRef	allocator,
					      CFStringRef	set,
					      CFStringRef	entity)
{
	/*
	 * create "/Sets/set-id/Network/Global/entity"
	 */
	return CFStringCreateWithFormat(allocator,
					NULL,
					CFSTR("/%@/%@/%@/%@/%@"),
					kSCPrefSets,
					set,
					kSCCompNetwork,
					kSCCompGlobal,
					entity);
}


CFStringRef
SCPreferencesPathKeyCreateSetNetworkInterfaceEntity(CFAllocatorRef	allocator,
						 CFStringRef	set,
						 CFStringRef	ifname,
						 CFStringRef	entity)
{
	/*
	 * create "/Sets/set-id/Network/Interface/interface-name/entity"
	 */
	if (entity == NULL) {
		return CFStringCreateWithFormat(allocator,
						NULL,
						CFSTR("/%@/%@/%@/%@/%@"),
						kSCPrefSets,
						set,
						kSCCompNetwork,
						kSCCompInterface,
						ifname);
	}
	return CFStringCreateWithFormat(allocator,
					NULL,
					CFSTR("/%@/%@/%@/%@/%@/%@"),
					kSCPrefSets,
					set,
					kSCCompNetwork,
					kSCCompInterface,
					ifname,
					entity);
}


CFStringRef
SCPreferencesPathKeyCreateSetNetworkService(CFAllocatorRef	allocator,
					    CFStringRef		set,
					    CFStringRef		service)
{
	CFStringRef path;

	if (service == NULL) {
		/*
		 * create "/Sets/set-id/Network/Service"
		 */
		path = CFStringCreateWithFormat(allocator,
						NULL,
						CFSTR("/%@/%@/%@/%@"),
						kSCPrefSets,
						set,
						kSCCompNetwork,
						kSCCompService);
	} else {
		/*
		 * create "/Sets/set-id/Network/Service/service-id"
		 */
		path = CFStringCreateWithFormat(allocator,
						NULL,
						CFSTR("/%@/%@/%@/%@/%@"),
						kSCPrefSets,
						set,
						kSCCompNetwork,
						kSCCompService,
						service);
	}

	return path;
}


CFStringRef
SCPreferencesPathKeyCreateSetNetworkServiceEntity(CFAllocatorRef	allocator,
						  CFStringRef		set,
						  CFStringRef		service,
						  CFStringRef		entity)
{
	CFStringRef path;

	if (entity == NULL) {
		/*
		 * create "/Sets/set-id/Network/Service/service-id"
		 */
		path = CFStringCreateWithFormat(allocator,
						NULL,
						CFSTR("/%@/%@/%@/%@/%@"),
						kSCPrefSets,
						set,
						kSCCompNetwork,
						kSCCompService,
						service);
	} else {
		/*
		 * create "/Sets/set-id/Network/Service/service-id/entity"
		 */
		path = CFStringCreateWithFormat(allocator,
						NULL,
						CFSTR("/%@/%@/%@/%@/%@/%@"),
						kSCPrefSets,
						set,
						kSCCompNetwork,
						kSCCompService,
						service,
						entity);
	}

	return path;
}

CFStringRef
SCPreferencesPathKeyCreateCategories(CFAllocatorRef allocator)
{
	/*
	 * path = "/Categories"
	 */
	return (CFStringCreateWithFormat(allocator,
					 NULL,
					 CFSTR("/%@"),
					 kSCPrefCategories));
}

CFStringRef
SCPreferencesPathKeyCreateCategory(CFAllocatorRef allocator,
				   CFStringRef category)
{
	/*
	 * path = "/Categories/<category>"
	 */
	return (CFStringCreateWithFormat(allocator,
					 NULL,
					 CFSTR("/%@/%@"),
					 kSCPrefCategories,
					 category));
}

#define kService	CFSTR("Service")

CFStringRef
SCPreferencesPathKeyCreateCategoryService(CFAllocatorRef allocator,
					  CFStringRef category,
					  CFStringRef value,
					  CFStringRef serviceID)
{
	CFStringRef	path;

	if (serviceID != NULL) {
		/*
		 * path = "/Categories/<category>/<value>/Service/<serviceID>"
		 */
		path = CFStringCreateWithFormat(allocator,
						NULL,
						CFSTR("/%@/%@/%@/%@/%@"),
						kSCPrefCategories,
						category,
						value,
						kService,
						serviceID);
	}
	else {
		/*
		 * path = "/Categories/<category>/<value>/Service"
		 */
		path = CFStringCreateWithFormat(allocator,
						NULL,
						CFSTR("/%@/%@/%@/%@"),
						kSCPrefCategories,
						category,
						value,
						kService);
	}
	return (path);
}

CFStringRef
SCPreferencesPathKeyCreateCategoryServiceEntity(CFAllocatorRef allocator,
						CFStringRef category,
						CFStringRef value,
						CFStringRef serviceID,
						CFStringRef entity)
{
	CFStringRef	path;

	/*
	 * path = "/Categories/<category>/<value>/Service/<serviceID>/<entity>"
	 */
	path = CFStringCreateWithFormat(allocator,
					NULL,
					CFSTR("/%@/%@/%@/%@/%@/%@"),
					kSCPrefCategories,
					category,
					value,
					kService,
					serviceID,
					entity);
	return (path);
}
