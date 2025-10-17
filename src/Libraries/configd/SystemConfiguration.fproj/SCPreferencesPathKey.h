/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#ifndef _SCPREFERENCESPATHKEY_H
#define _SCPREFERENCESPATHKEY_H

#include <os/availability.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>

/*!
	@header SCPreferencesPathKey
 */

__BEGIN_DECLS

/*
 * SCPreferencesPathKeyCreate*
 * - convenience routines that create a CFString key for an item in the store
 */

/*!
	@function SCPreferencesPathKeyCreate
	@discussion Creates a preferences path key using the given format.
 */
CFStringRef
SCPreferencesPathKeyCreate			(
						CFAllocatorRef	allocator,
						CFStringRef	fmt,
						...
						)	CF_FORMAT_FUNCTION(2,3)	API_AVAILABLE(macos(10.4), ios(2.0));

/*!
	@function SCPreferencesPathKeyCreateNetworkServices
 */
CFStringRef
SCPreferencesPathKeyCreateNetworkServices	(
						CFAllocatorRef	allocator
						)				API_AVAILABLE(macos(10.4), ios(2.0));

/*!
	@function SCPreferencesPathKeyCreateNetworkServiceEntity
 */
CFStringRef
SCPreferencesPathKeyCreateNetworkServiceEntity	(
						CFAllocatorRef	allocator,
						CFStringRef	service,
						CFStringRef	entity
						)				API_AVAILABLE(macos(10.4), ios(2.0));

/*!
	@function SCPreferencesPathKeyCreateSets
 */
CFStringRef
SCPreferencesPathKeyCreateSets			(
						CFAllocatorRef	allocator
						)				API_AVAILABLE(macos(10.4), ios(2.0));

/*!
	@function SCPreferencesPathKeyCreateSet
 */
CFStringRef
SCPreferencesPathKeyCreateSet			(
						CFAllocatorRef	allocator,
						CFStringRef	set
						)				API_AVAILABLE(macos(10.4), ios(2.0));

/*!
	@function SCPreferencesPathKeyCreateSetNetworkInterfaceEntity
 */
CFStringRef
SCPreferencesPathKeyCreateSetNetworkInterfaceEntity(
						   CFAllocatorRef	allocator,
						   CFStringRef	set,
						   CFStringRef	ifname,
						   CFStringRef	entity
						   )				API_AVAILABLE(macos(10.4), ios(2.0));

/*!
	@function SCPreferencesPathKeyCreateSetNetworkGlobalEntity
 */
CFStringRef
SCPreferencesPathKeyCreateSetNetworkGlobalEntity(
						CFAllocatorRef	allocator,
						CFStringRef	set,
						CFStringRef	entity
						)				API_AVAILABLE(macos(10.4), ios(2.0));

/*!
	@function SCPreferencesPathKeyCreateSetNetworkService
 */
CFStringRef
SCPreferencesPathKeyCreateSetNetworkService     (
						CFAllocatorRef	allocator,
						CFStringRef	set,
						CFStringRef	service
						)				API_AVAILABLE(macos(10.4), ios(2.0));

/*!
	@function SCPreferencesPathKeyCreateSetNetworkServiceEntity
 */
CFStringRef
SCPreferencesPathKeyCreateSetNetworkServiceEntity(
						 CFAllocatorRef	allocator,
						 CFStringRef	set,
						 CFStringRef	service,
						 CFStringRef	entity
						 )				API_AVAILABLE(macos(10.4), ios(2.0));


/*!
	@function SCPreferencesPathKeyCreateCategories
 */
CFStringRef
SCPreferencesPathKeyCreateCategories(CFAllocatorRef allocator)
	API_AVAILABLE(macos(14.0), ios(17.0));

/*!
	@function SCPreferencesPathKeyCreateCategory
 */
CFStringRef
SCPreferencesPathKeyCreateCategory(CFAllocatorRef allocator,
				   CFStringRef category)
	API_AVAILABLE(macos(14.0), ios(17.0));

CFStringRef
SCPreferencesPathKeyCreateCategoryService(CFAllocatorRef allocator,
					  CFStringRef category,
					  CFStringRef value,
					  CFStringRef serviceID)
	API_AVAILABLE(macos(14.0), ios(17.0));

/*!
	@function SCPreferencesPathKeyCreateCategoryServiceEntity
 */
CFStringRef
SCPreferencesPathKeyCreateCategoryServiceEntity(CFAllocatorRef allocator,
						CFStringRef category,
						CFStringRef value,
						CFStringRef serviceID,
						CFStringRef entity)
	API_AVAILABLE(macos(14.0), ios(17.0));

__END_DECLS

#endif	/* _SCPREFERENCESPATHKEY_H */
