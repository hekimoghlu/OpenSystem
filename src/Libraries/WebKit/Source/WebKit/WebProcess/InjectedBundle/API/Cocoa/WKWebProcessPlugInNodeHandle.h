/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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
#import <JavaScriptCore/JavaScriptCore.h>
#import <WebKit/WKImage.h>

#if TARGET_OS_IPHONE
#import <UIKit/UIImage.h>
#endif

@class WKWebProcessPlugInFrame;

WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKWebProcessPlugInNodeHandle : NSObject

+ (WKWebProcessPlugInNodeHandle *)nodeHandleWithJSValue:(JSValue *)value inContext:(JSContext *)context;
- (WKWebProcessPlugInFrame *)htmlIFrameElementContentFrame;

#if TARGET_OS_IPHONE
- (UIImage *)renderedImageWithOptions:(WKSnapshotOptions)options WK_API_AVAILABLE(ios(9.0));
- (UIImage *)renderedImageWithOptions:(WKSnapshotOptions)options width:(NSNumber *)width WK_API_AVAILABLE(ios(11.0));
#else
- (NSImage *)renderedImageWithOptions:(WKSnapshotOptions)options WK_API_AVAILABLE(macos(10.11));
- (NSImage *)renderedImageWithOptions:(WKSnapshotOptions)options width:(NSNumber *)width WK_API_AVAILABLE(macos(10.13));
#endif

@property (nonatomic, readonly) CGRect elementBounds;
@property (nonatomic) BOOL HTMLInputElementIsAutoFilled;
@property (nonatomic) BOOL HTMLInputElementIsAutoFilledAndViewable;
@property (nonatomic) BOOL HTMLInputElementIsAutoFilledAndObscured WK_API_AVAILABLE(macos(13.0), ios(16.0));
@property (nonatomic, readonly) BOOL HTMLInputElementIsUserEdited;
@property (nonatomic, readonly) BOOL HTMLTextAreaElementIsUserEdited;
@property (nonatomic, readonly) WKWebProcessPlugInNodeHandle *HTMLTableCellElementCellAbove;
@property (nonatomic, readonly) WKWebProcessPlugInFrame *frame;
@property (nonatomic, readonly) BOOL isSelectElement WK_API_AVAILABLE(macos(10.14.4), ios(12.2));
@property (nonatomic, readonly) BOOL isSelectableTextNode WK_API_AVAILABLE(macos(12.0), ios(15.0));

- (BOOL)isTextField;

@end
