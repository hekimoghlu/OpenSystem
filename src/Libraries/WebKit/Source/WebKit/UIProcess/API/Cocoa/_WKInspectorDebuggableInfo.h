/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#import <WebKit/WKFoundation.h>

/*! A _WKInspectorDebuggableInfo object contains information about a debuggable target.
 @discussion An instance of this class is a transient, data-only object;
 it does not uniquely identify a debuggable across multiple method calls.
 */

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, _WKInspectorDebuggableType) {
    _WKInspectorDebuggableTypeITML,
    _WKInspectorDebuggableTypeJavaScript,
    _WKInspectorDebuggableTypeServiceWorker,
    _WKInspectorDebuggableTypePage,
    _WKInspectorDebuggableTypeWebPage,
} WK_API_AVAILABLE(macos(10.15.4), ios(13.4));

WK_CLASS_AVAILABLE(macos(10.15.4), ios(13.4))
@interface _WKInspectorDebuggableInfo : NSObject <NSCopying>
@property (nonatomic) _WKInspectorDebuggableType debuggableType;
@property (nonatomic, copy) NSString *targetPlatformName;
@property (nonatomic, copy) NSString *targetBuildVersion;
@property (nonatomic, copy) NSString *targetProductVersion;
@property (nonatomic) BOOL targetIsSimulator;
@end

NS_ASSUME_NONNULL_END
