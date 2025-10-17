/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

NS_ASSUME_NONNULL_BEGIN

/*! @abstract A WKContentWorldConfiguration object allows you to specify configuration for WKContentWorld.
@discussion WKContentWorldConfiguration allows applications to specify ways by which extra JavaScript capabilities should be exposed to the script in the environment.
For example:
- If your scripts have to access autofill capabilities, you may want to set allowAutofill to YES. */
WK_SWIFT_UI_ACTOR
NS_SWIFT_SENDABLE
WK_CLASS_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
@interface _WKContentWorldConfiguration : NSObject<NSCopying, NSSecureCoding>

@property (nonatomic, copy) NSString *name;

/*! @abstract A boolean value indicating whether every shadow root should be treated as open mode shadow root or not. */
@property (nonatomic) BOOL allowAccessToClosedShadowRoots;

/*! @abstract A boolean value indicating whether the capability to trigger autofill is exposed to scripts or not. */
@property (nonatomic) BOOL allowAutofill;

/*! @abstract A boolean value indicating whether the ability to attach user info on an element is exposed to scripts or not. */
@property (nonatomic) BOOL allowElementUserInfo;

/*! @abstract A boolean value indicating whether the behavior that elements with a name attribute overrides builtin methods on document object should be disabled or not. */
@property (nonatomic) BOOL disableLegacyBuiltinOverrides;

@end

NS_ASSUME_NONNULL_END
