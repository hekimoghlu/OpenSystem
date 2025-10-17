/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
//  IOHIDPreferences.h
//  IOKitUser
//
//  Created by AB on 10/25/19.
//

#ifndef IOHIDPreferences_h
#define IOHIDPreferences_h

#include <CoreFoundation/CoreFoundation.h>

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

typedef enum {
    IOHIDPreferencesOptionNone,
    IOHIDPreferencesOptionSingletonConnection
} IOHIDPreferencesOption;

/*! These APIs are XPC wrapper around corresponding CFPreferences API. https://developer.apple.com/documentation/corefoundation/preferences_utilities
 * XPC HID Preferences APIs are currently available for macOS only .
 */
CF_EXPORT
void IOHIDPreferencesSet(CFStringRef key, CFTypeRef __nullable value, CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
void IOHIDPreferencesSetMultiple(CFDictionaryRef __nullable keysToSet , CFArrayRef __nullable keysToRemove, CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
CFTypeRef __nullable IOHIDPreferencesCopy(CFStringRef key, CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
CFDictionaryRef __nullable IOHIDPreferencesCopyMultiple(CFArrayRef __nullable keys, CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
void IOHIDPreferencesSynchronize(CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
CFTypeRef __nullable IOHIDPreferencesCopyDomain(CFStringRef key, CFStringRef domain);

CF_EXPORT
void IOHIDPreferencesSetDomain(CFStringRef key,  CFTypeRef __nullable value, CFStringRef domain);

#pragma mark -
#pragma mark -

CF_EXPORT
CFTypeRef __nullable IOHIDPreferencesCreateInstance(IOHIDPreferencesOption option);

CF_EXPORT
void IOHIDPreferencesSetForInstance(CFTypeRef hidPreference, CFStringRef key, CFTypeRef __nullable value, CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
void IOHIDPreferencesSetMultipleForInstance(CFTypeRef hidPreference, CFDictionaryRef __nullable keysToSet , CFArrayRef __nullable keysToRemove, CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
CFTypeRef __nullable IOHIDPreferencesCopyForInstance(CFTypeRef hidPreference, CFStringRef key, CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
CFDictionaryRef __nullable IOHIDPreferencesCopyMultipleForInstance(CFTypeRef hidPreference, CFArrayRef __nullable keys, CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
void IOHIDPreferencesSynchronizeForInstance(CFTypeRef hidPreference, CFStringRef user, CFStringRef host, CFStringRef domain);

CF_EXPORT
CFTypeRef __nullable IOHIDPreferencesCopyDomainForInstance(CFTypeRef hidPreference, CFStringRef key, CFStringRef domain);

CF_EXPORT
void IOHIDPreferencesSetDomainForInstance(CFTypeRef hidPreference, CFStringRef key,  CFTypeRef __nullable value, CFStringRef domain);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS

#endif /* IOHIDPreferences_h */
