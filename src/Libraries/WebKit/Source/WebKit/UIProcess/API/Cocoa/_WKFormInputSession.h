/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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

#import <Foundation/Foundation.h>
#import <WebKit/_WKFocusedElementInfo.h>

@class UITextSuggestion;

@protocol _WKFormInputSession <NSObject>

@property (nonatomic, readonly, getter=isValid) BOOL valid;
@property (nonatomic, readonly) NSObject <NSSecureCoding> *userObject;
@property (nonatomic, readonly) id <_WKFocusedElementInfo> focusedElementInfo WK_API_AVAILABLE(macos(10.12), ios(10.0));

#if TARGET_OS_IPHONE
@property (nonatomic, copy) NSString *accessoryViewCustomButtonTitle;
@property (nonatomic, strong) UIView *customInputView WK_API_AVAILABLE(ios(10.0));
@property (nonatomic, strong) UIView *customInputAccessoryView WK_API_AVAILABLE(ios(12.2));
@property (nonatomic, copy) NSArray<UITextSuggestion *> *suggestions WK_API_AVAILABLE(ios(10.0));
@property (nonatomic) BOOL accessoryViewShouldNotShow WK_API_AVAILABLE(ios(10.0));
@property (nonatomic) BOOL forceSecureTextEntry WK_API_AVAILABLE(ios(10.0));
@property (nonatomic, readonly) BOOL requiresStrongPasswordAssistance WK_API_AVAILABLE(ios(12.0));

- (void)reloadFocusedElementContextView WK_API_AVAILABLE(ios(12.0));
#endif

@end
