/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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

#include <WebCore/BackgroundFetchFailureReason.h>
#include <WebCore/BackgroundFetchOptions.h>
#include <WebCore/BackgroundFetchResult.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/URL.h>

#if PLATFORM(COCOA)
OBJC_CLASS NSDictionary;
#endif

namespace WebKit {

struct BackgroundFetchState {
    WebCore::SecurityOriginData topOrigin;
    URL scope;
    String identifier;
    
    WebCore::BackgroundFetchOptions options;
    
    uint64_t downloadTotal { 0 };
    uint64_t downloaded { 0 };
    uint64_t uploadTotal { 0 };
    uint64_t uploaded { 0 };
    
    WebCore::BackgroundFetchResult result { WebCore::BackgroundFetchResult::EmptyString };
    WebCore::BackgroundFetchFailureReason failureReason { WebCore::BackgroundFetchFailureReason::EmptyString };
    
    bool isPaused { false };
    
#if PLATFORM(COCOA)
    NSDictionary *toDictionary() const;
#endif
};

} // namespace WebKit
