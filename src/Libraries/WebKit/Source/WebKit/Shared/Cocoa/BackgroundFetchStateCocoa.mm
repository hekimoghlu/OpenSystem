/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
#import "BackgroundFetchState.h"

namespace WebCore {
String convertEnumerationToString(BackgroundFetchResult);
String convertEnumerationToString(BackgroundFetchFailureReason);
}

namespace WebKit {

NSDictionary *BackgroundFetchState::toDictionary() const
{
    // FIXME: Expose icon URLS.
    return @{
        @"TopOrigin" : (NSString *)topOrigin.toString(),
        @"Scope" : (NSURL *)scope,
        @"WebIdentifier" : (NSString *)identifier,
        @"Title" : (NSString *)options.title,
        @"DownloadTotal" : @(downloadTotal),
        @"Downloaded" : @(downloaded),
        @"UploadTotal" : @(uploadTotal),
        @"Uploaded" : @(uploaded),
        @"Result" : (NSString *)(convertEnumerationToString(result)),
        @"FailureReason" : (NSString *)(convertEnumerationToString(failureReason)),
        @"IsPaused" : @(isPaused),
    };
}

} // namespace WebKit
