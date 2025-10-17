/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#pragma once

#if defined(__OBJC__)
#import <Foundation/NSURLCache.h>
#endif

namespace WebKit {

enum class CacheStoragePolicy : uint8_t {
    Allowed = 0,
    AllowedInMemoryOnly,
    NotAllowed
};

#if defined(__OBJC__)

inline NSURLCacheStoragePolicy toNSURLCacheStoragePolicy(CacheStoragePolicy policy)
{
    switch (policy) {
    case CacheStoragePolicy::Allowed:
        return NSURLCacheStorageAllowed;
    case CacheStoragePolicy::AllowedInMemoryOnly:
        return NSURLCacheStorageAllowedInMemoryOnly;
    case CacheStoragePolicy::NotAllowed:
        return NSURLCacheStorageNotAllowed;
    }
    ASSERT_NOT_REACHED();
    return NSURLCacheStorageNotAllowed;
}

inline CacheStoragePolicy toCacheStoragePolicy(NSURLCacheStoragePolicy policy)
{
    switch (policy) {
    case NSURLCacheStorageAllowed:
        return CacheStoragePolicy::Allowed;
    case NSURLCacheStorageAllowedInMemoryOnly:
        return CacheStoragePolicy::AllowedInMemoryOnly;
    case NSURLCacheStorageNotAllowed:
        return CacheStoragePolicy::NotAllowed;
    }
    ASSERT_NOT_REACHED();
    return CacheStoragePolicy::NotAllowed;
}

#endif // defined(__OBJC__)

} // namespace WebKit
