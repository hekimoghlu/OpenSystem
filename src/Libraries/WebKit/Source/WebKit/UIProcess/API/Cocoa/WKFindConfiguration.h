/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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

WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.15.4), ios(13.4))
@interface WKFindConfiguration : NSObject <NSCopying>

/* @abstract The direction to search from the current selection
 @discussion The search will respect the writing direction of the document.
 The initial value is NO.
*/
@property (nonatomic) BOOL backwards;

/* @abstract Whether or not the search should be case sensitive
 @discussion The initial value is NO.
*/
@property (nonatomic) BOOL caseSensitive;

/* @abstract Whether the search should start at the beginning of the document once it reaches the end
 @discussion The initial value is YES.
*/
@property (nonatomic) BOOL wraps;

@end

NS_ASSUME_NONNULL_END
