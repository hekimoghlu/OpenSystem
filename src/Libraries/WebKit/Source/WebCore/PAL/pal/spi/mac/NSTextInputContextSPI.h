/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#if PLATFORM(MAC)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSTextInputContext_Private.h>
#import <AppKit/NSTextPlaceholder_Private.h>

#else // !USE(APPLE_INTERNAL_SDK)

@interface NSTextSelectionRect : NSObject
@property (nonatomic, readonly) NSRect rect;
@property (nonatomic, readonly) NSWritingDirection writingDirection;
@property (nonatomic, readonly) BOOL isVertical;
@property (nonatomic, readonly) NSAffineTransform *transform;
@end

@interface NSTextPlaceholder : NSObject
@property (nonatomic, readonly) NSArray<NSTextSelectionRect *> *rects;
@end

@interface NSTextInputContext ()
- (void)handleEvent:(NSEvent *)event completionHandler:(void(^)(BOOL handled))completionHandler;
- (void)handleEventByInputMethod:(NSEvent *)event completionHandler:(void(^)(BOOL handled))completionHandler;
- (BOOL)handleEventByKeyboardLayout:(NSEvent *)event;

#if HAVE(REDESIGNED_TEXT_CURSOR)
@property BOOL showsCursorAccessories;
#endif
@end

#endif // USE(APPLE_INTERNAL_SDK)

APPKIT_EXTERN NSString *NSTextInsertionUndoableAttributeName;
APPKIT_EXTERN NSString *NSTextInputReplacementRangeAttributeName;

#endif // PLATFORM(MAC)
