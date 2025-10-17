/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#import <wtf/Platform.h>

#if PLATFORM(MAC)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSMenu_Private.h>

@interface NSMenuItem (Staging_129192954)

+ (NSMenuItem *)standardWritingToolsMenuItem;

@end

#elif USE(APPLE_INTERNAL_SDK)

WTF_EXTERN_C_BEGIN
#import <AppKit/NSMenu_Private.h>
WTF_EXTERN_C_END

#else

typedef NS_ENUM(NSInteger, NSMenuType) {
    NSMenuTypeNone = 0,
    NSMenuTypeContextMenu,
};

enum {
    NSPopUpMenuTypeGeneric,
    NSPopUpMenuTypePopUp,
    NSPopUpMenuTypePullsDown,
    NSPopUpMenuTypeMainMenu,
    NSPopUpMenuTypeContext,
    NSPopUpMenuDefaultToPopUpControlFont = 0x10,
    NSPopUpMenuPositionRelativeToRightEdge = 0x20,
    NSPopUpMenuIsPopupButton = 0x40,
};

@interface NSMenu ()
+ (NSMenuType)menuTypeForEvent:(NSEvent *)event;
@end

@class QLPreviewMenuItem;

@interface NSMenuItem ()
+ (QLPreviewMenuItem *)standardQuickLookMenuItem;
+ (NSMenuItem *)standardShareMenuItemForItems:(NSArray *)items;
+ (NSMenuItem *)standardWritingToolsMenuItem;
@end

#endif

@interface NSMenu (Staging_81123724)
- (BOOL)_containsItemMatchingEvent:(NSEvent *)event includingDisabledItems:(BOOL)includingDisabledItems;
@end

#if ENABLE(CONTEXT_MENU_IMAGES_FOR_INTERNAL_CLIENTS)
@interface NSMenuItem (Staging_138651669)

+ (NSString *)_systemImageNameForAction:(SEL)action;
@property (strong, setter=_setActionImage:) NSImage *_actionImage;
@property (setter=_setHasActionImage:) BOOL _hasActionImage;

@end
#endif

typedef NSUInteger NSPopUpMenuFlags;

WTF_EXTERN_C_BEGIN

extern NSString * const NSPopUpMenuPopupButtonBounds;
extern NSString * const NSPopUpMenuPopupButtonLabelOffset;
extern NSString * const NSPopUpMenuPopupButtonSize;
extern NSString * const NSPopUpMenuPopupButtonWidget;

void _NSPopUpCarbonMenu3(NSMenu *, NSWindow *, NSView *ownerView, NSPoint screenPoint, NSInteger checkedItem, NSFont *, CGFloat minWidth, NSString *runLoopMode, NSPopUpMenuFlags, NSDictionary *options);

WTF_EXTERN_C_END

#endif // PLATFORM(MAC)
