/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#ifndef _SCNETWORKCATEGORY_H
#define _SCNETWORKCATEGORY_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <unistd.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCNetworkCategoryTypes.h>

CF_IMPLICIT_BRIDGING_ENABLED
CF_ASSUME_NONNULL_BEGIN

/*!
	@header SCNetworkCategory
 */

__BEGIN_DECLS

typedef struct CF_BRIDGED_TYPE(id) __SCNetworkCategory * SCNetworkCategoryRef;


CFTypeID
SCNetworkCategoryGetTypeID(void)
	API_AVAILABLE(macos(14.0), ios(17.0));	

CFArrayRef __nullable /* of SCNetworkCategoryRef */
SCNetworkCategoryCopyAll(SCPreferencesRef prefs)
	API_AVAILABLE(macos(14.0), ios(17.0));	

SCNetworkCategoryRef __nullable
SCNetworkCategoryCreate(SCPreferencesRef prefs,
			CFStringRef category)
	API_AVAILABLE(macos(14.0), ios(17.0));	

Boolean
SCNetworkCategoryAddService(SCNetworkCategoryRef category,
			    CFStringRef value,
			    SCNetworkServiceRef service)
	API_AVAILABLE(macos(14.0), ios(17.0));	

Boolean
SCNetworkCategoryRemoveService(SCNetworkCategoryRef category,
			       CFStringRef value,
			       SCNetworkServiceRef service)
	API_AVAILABLE(macos(14.0), ios(17.0));	


CFArrayRef __nullable /* of SCNetworkServiceRef */
SCNetworkCategoryCopyServices(SCNetworkCategoryRef category,
			      CFStringRef value)
	API_AVAILABLE(macos(14.0), ios(17.0));	

CFArrayRef __nullable /* of CFStringRef category value */
SCNetworkCategoryCopyValues(SCNetworkCategoryRef category)
	API_AVAILABLE(macos(14.0), ios(17.0));	

Boolean
SCNetworkCategorySetServiceQoSMarkingPolicy(SCNetworkCategoryRef category,
					    CFStringRef value,
					    SCNetworkServiceRef service,
					    CFDictionaryRef __nullable entity)
	API_AVAILABLE(macos(14.0), ios(17.0));

CFDictionaryRef __nullable
SCNetworkCategoryGetServiceQoSMarkingPolicy(SCNetworkCategoryRef category,
					    CFStringRef value,
					    SCNetworkServiceRef service)
	API_AVAILABLE(macos(14.0), ios(17.0));

__END_DECLS

CF_ASSUME_NONNULL_END
CF_IMPLICIT_BRIDGING_DISABLED

#endif	/* _SCNETWORKCATEGORY_H */
