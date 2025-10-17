/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
 * SOSPersist.h -- Utility routines for get/set in CFDictionary
 */

#ifndef _SOSPERSIST_H_
#define _SOSPERSIST_H_

__BEGIN_DECLS

#include <utilities/SecCFRelease.h>
#include <utilities/SecCFWrappers.h>
#include <CoreFoundation/CoreFoundation.h>

#include <stdlib.h>

#include <AssertMacros.h>


static inline bool SOSPeerGetPersistedBoolean(CFDictionaryRef persisted, CFStringRef key) {
    CFBooleanRef boolean = CFDictionaryGetValue(persisted, key);
    return boolean && CFBooleanGetValue(boolean);
}

static inline CFDataRef SOSPeerGetPersistedData(CFDictionaryRef persisted, CFStringRef key) {
    return asData(CFDictionaryGetValue(persisted, key), NULL);
}

static inline int64_t SOSPeerGetPersistedInt64(CFDictionaryRef persisted, CFStringRef key) {
    int64_t integer = 0;
    CFNumberRef number = CFDictionaryGetValue(persisted, key);
    if (number) {
        CFNumberGetValue(number, kCFNumberSInt64Type, &integer);
    }
    return integer;
}

static inline bool SOSPeerGetOptionalPersistedCFIndex(CFDictionaryRef persisted, CFStringRef key, CFIndex *value) {
    bool exists = false;
    CFNumberRef number = CFDictionaryGetValue(persisted, key);
    if (number) {
        exists = true;
        CFNumberGetValue(number, kCFNumberCFIndexType, value);
    }
    return exists;
}

static inline void SOSPersistBool(CFMutableDictionaryRef persist, CFStringRef key, bool value) {
    CFDictionarySetValue(persist, key, value ? kCFBooleanTrue : kCFBooleanFalse);
}

static inline void SOSPersistInt64(CFMutableDictionaryRef persist, CFStringRef key, int64_t value) {
    CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &value);
    CFDictionarySetValue(persist, key, number);
    CFReleaseSafe(number);
}

static inline void SOSPersistCFIndex(CFMutableDictionaryRef persist, CFStringRef key, CFIndex value) {
    CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberCFIndexType, &value);
    CFDictionarySetValue(persist, key, number);
    CFReleaseSafe(number);
}

static inline void SOSPersistOptionalValue(CFMutableDictionaryRef persist, CFStringRef key, CFTypeRef value) {
    if (value)
        CFDictionarySetValue(persist, key, value);
}

__END_DECLS

#endif /* !_SOSPERSIST_H_ */
