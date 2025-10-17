/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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
#ifndef _CATEGORY_MANAGER_INTERNAL_H
#define _CATEGORY_MANAGER_INTERNAL_H

#include <SystemConfiguration/SCPrivate.h>

#define	my_log(__level, __format, ...)	SC_log(__level, __format, ## __VA_ARGS__)

#define kNetworkCategoryManagerServerName \
    "com.apple.SystemConfiguration.NetworkCategoryManager"

typedef CF_ENUM(uint32_t, CategoryManagerRequestType) {
    kCategoryManagerRequestTypeNone			= 0,
    kCategoryManagerRequestTypeRegister			= 1,
    kCategoryManagerRequestTypeActivateValue		= 2,
    kCategoryManagerRequestTypeGetActiveValue		= 3,
};


/*
 * kCategoryManagerRequestKey*
 * - keys used to communicate a request to the server
 */
#define kCategoryManagerRequestKeyType			"Type"
#define kCategoryManagerRequestKeyProcessName		"ProcessName"
#define kCategoryManagerRequestKeyCategory		"Category"
#define kCategoryManagerRequestKeyInterfaceName		"InterfaceName"
#define kCategoryManagerRequestKeyFlags			"Flags"
#define kCategoryManagerRequestKeyValue			"Value"

/*
 * kCategoryManagerResponseKey*
 * - keys used to communicate the response from the server
 */
#define kCategoryManagerResponseKeyError		"Error"
#define kCategoryManagerResponseKeyActiveValue		"ActiveValue"

/*
 * kCategoryInformationKey*
 * - keys used in CategoryInformationRef dictionary
 */
#define kCategoryInformationKeyCategory			\
	CFSTR(kCategoryManagerRequestKeyCategory)
#define kCategoryInformationKeyInterfaceName		\
	CFSTR(kCategoryManagerRequestKeyInterfaceName)
#define kCategoryInformationKeyValue		\
	CFSTR(kCategoryManagerRequestKeyValue)
#define kCategoryInformationKeyFlags		\
	CFSTR(kCategoryManagerRequestKeyFlags)
#define kCategoryInformationKeyProcessName			\
	CFSTR(kCategoryManagerRequestKeyProcessName)
#define kCategoryInformationKeyActiveValue		CFSTR("ActiveValue")
#define kCategoryInformationKeyProcessID		CFSTR("ProcessID")


static inline CFStringRef
cfstring_create_with_cstring(const char * str)
{
	if (str == NULL) {
		return (NULL);
	}
	return (CFStringCreateWithCString(NULL, str, kCFStringEncodingUTF8));
}

static inline char *
cstring_create_with_cfstring(CFStringRef str)
{
	if (str == NULL) {
		return (NULL);
	}
	return ( _SC_cfstring_to_cstring(str, NULL, 0,
					 kCFStringEncodingUTF8));
}

static inline void
cstring_deallocate(char * str)
{
	if (str == NULL) {
		return;
	}
	CFAllocatorDeallocate(NULL, str);
}

static inline void
my_xpc_dictionary_set_cfstring(xpc_object_t dict, char * key,
			       CFStringRef value)
{
	char *	value_c;

	value_c = cstring_create_with_cfstring(value);
	xpc_dictionary_set_string(dict, key, value_c);
	cstring_deallocate(value_c);
}

#endif /* _CATEGORY_MANAGER_INTERNAL_H */
