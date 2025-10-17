/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#import <Quartz/Quartz.h>

#if USE(APPLE_INTERNAL_SDK)

#import <Quartz/QuartzPrivate.h>

#else

@protocol QLPreviewMenuItemDelegate <NSObject>
@optional

- (NSView *)menuItem:(NSMenuItem *)menuItem viewAtScreenPoint:(NSPoint)screenPoint;
- (id<QLPreviewItem>)menuItem:(NSMenuItem *)menuItem previewItemAtPoint:(NSPoint)point;
- (NSRectEdge)menuItem:(NSMenuItem *)menuItem preferredEdgeForPoint:(NSPoint)point;
- (void)menuItemDidClose:(NSMenuItem *)menuItem;
- (NSRect)menuItem:(NSMenuItem *)menuItem itemFrameForPoint:(NSPoint)point;
- (NSSize)menuItem:(NSMenuItem *)menuItem maxSizeForPoint:(NSPoint)point;

@end

@interface QLPreviewMenuItem : NSObject
@end

@interface QLPreviewMenuItem ()
typedef NS_ENUM(NSInteger, QLPreviewStyle) {
    QLPreviewStyleStandaloneWindow,
    QLPreviewStylePopover
};

- (void)close;

@property (assign) id<QLPreviewMenuItemDelegate> delegate;
@property QLPreviewStyle previewStyle;
@end

@interface QLItem : NSObject <QLPreviewItem>
@property (nonatomic, copy) NSDictionary* previewOptions;
@end

#if HAVE(QUICKLOOK_PREVIEW_ACTIVITY)

typedef NS_ENUM(NSInteger, QLPreviewActivity) {
    QLPreviewActivityNone,
    QLPreviewActivityMarkup,
    QLPreviewActivityTrim,
    QLPreviewActivityVisualSearch
};

#endif // HAVE(QUICKLOOK_PREVIEW_ACTIVITY)

#endif // USE(APPLE_INTERNAL_SDK)

#if HAVE(QUICKLOOK_PREVIEW_ITEM_DATA_PROVIDER)

@class UTType;
@interface QLItem (Staging_74299451)
- (instancetype)initWithDataProvider:(id /* <QLPreviewItemDataProvider> */)data contentType:(UTType *)contentType previewTitle:(NSString *)previewTitle;
@end

#endif // HAVE(QUICKLOOK_PREVIEW_ITEM_DATA_PROVIDER)

#if HAVE(QUICKLOOK_ITEM_PREVIEW_OPTIONS)

@interface QLItem (Staging_77864637)
@property (nonatomic, copy) NSDictionary *previewOptions;
@end

#endif // HAVE(QUICKLOOK_ITEM_PREVIEW_OPTIONS)
