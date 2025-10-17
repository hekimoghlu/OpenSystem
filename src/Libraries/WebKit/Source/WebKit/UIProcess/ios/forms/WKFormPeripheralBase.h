/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
#if PLATFORM(IOS_FAMILY)

#import "WKFormPeripheral.h"
#import <wtf/Forward.h>
#import <wtf/RetainPtr.h>

@class WKContentView;

@interface WKFormPeripheralBase : NSObject <WKFormPeripheral>

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithView:(WKContentView *)view control:(RetainPtr<NSObject <WKFormControl>>&&)control NS_DESIGNATED_INITIALIZER;

- (void)beginEditing;
- (void)updateEditing;
- (void)endEditing;
- (UIView *)assistantView;
- (BOOL)handleKeyEvent:(UIEvent *)event;

@property (nonatomic, readonly) WKContentView *view;
@property (nonatomic, readonly) NSObject <WKFormControl> *control;
@property (nonatomic, readonly, getter=isEditing) BOOL editing;
@property (nonatomic) BOOL singleTapShouldEndEditing;

@end

#endif
