/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

#if PLATFORM(MAC)
#import <AppKit/AppKit.h>
#else
#import <UIKit/UIKit.h>
#endif

#if HAVE(TRANSLATION_UI_SERVICES)

#if USE(APPLE_INTERNAL_SDK)
#import <TranslationUIServices/LTUISourceMeta.h>
#import <TranslationUIServices/LTUITranslationViewController.h>
#else
typedef NS_ENUM(NSUInteger, LTUISourceMetaOrigin) {
    LTUISourceMetaOriginUnspecified,
    LTUISourceMetaOriginImage
};

@interface LTUISourceMeta : NSObject
@property (nonatomic) LTUISourceMetaOrigin origin;
@end

#if PLATFORM(MAC)
@interface LTUITranslationViewController : NSViewController
#else
@interface LTUITranslationViewController : UIViewController
#endif
@property (class, readonly, getter=isAvailable) BOOL available;
@property (nonatomic, copy) NSAttributedString *text;
@property (nonatomic) BOOL isSourceEditable;
@property (nonatomic, strong) LTUISourceMeta *sourceMeta;
@property (nonatomic, copy) void(^replacementHandler)(NSAttributedString *);
@end
#endif

#endif // HAVE(TRANSLATION_UI_SERVICES)
