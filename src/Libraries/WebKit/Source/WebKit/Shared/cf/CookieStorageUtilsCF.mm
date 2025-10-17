/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
#import "config.h"
#import "CookieStorageUtilsCF.h"

#import <pal/spi/cocoa/NSURLConnectionSPI.h>
#import <wtf/ProcessPrivilege.h>
#import <wtf/cf/VectorCF.h>

namespace WebKit {

RetainPtr<CFHTTPCookieStorageRef> cookieStorageFromIdentifyingData(const Vector<uint8_t>& data)
{
    ASSERT(!data.isEmpty());
    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies));

    auto cookieStorageData = adoptCF(CFDataCreate(kCFAllocatorDefault, data.data(), data.size()));
    auto cookieStorage = adoptCF(CFHTTPCookieStorageCreateFromIdentifyingData(kCFAllocatorDefault, cookieStorageData.get()));
    ASSERT(cookieStorage);

    CFHTTPCookieStorageScheduleWithRunLoop(cookieStorage.get(), [NSURLConnection resourceLoaderRunLoop], kCFRunLoopCommonModes);

    return cookieStorage;
}

Vector<uint8_t> identifyingDataFromCookieStorage(CFHTTPCookieStorageRef cookieStorage)
{
    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies));
    return makeVector(adoptCF(CFHTTPCookieStorageCreateIdentifyingData(kCFAllocatorDefault, cookieStorage)).get());
}

} // namespace WebKit
