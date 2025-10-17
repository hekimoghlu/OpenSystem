/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#import <AppKit/AppKit.h>

#if USE(APPLE_INTERNAL_SDK)

#define CGCOLORTAGGEDPOINTER_H_

#import <AppKit/NSInspectorBar.h>
#import <AppKit/NSMenu_Private.h>
#import <AppKit/NSPreviewRepresentingActivityItem_Private.h>
#import <AppKit/NSTextInputClient_Private.h>
#import <AppKit/NSWindow_Private.h>

#if HAVE(NSSCROLLVIEW_SEPARATOR_TRACKING_ADAPTER)
#import <AppKit/NSScrollViewSeparatorTrackingAdapter_Private.h>
#endif

#else

@interface NSInspectorBar : NSObject
@property (getter=isVisible) BOOL visible;
@end

@interface NSKeyboardShortcut
+ (id)shortcutWithKeyEquivalent:(NSString *)keyEquivalent modifierMask:(NSUInteger)modifierMask;
@property (readonly) NSString *localizedDisplayName;
@end

#if HAVE(NSSCROLLVIEW_SEPARATOR_TRACKING_ADAPTER)
@protocol NSScrollViewSeparatorTrackingAdapter
@property (readonly) NSRect scrollViewFrame;
@property (readonly) BOOL hasScrolledContentsUnderTitlebar;
@end
#endif

@protocol NSTextInputClient_Async
@end

typedef NS_OPTIONS(NSUInteger, NSWindowShadowOptions) {
    NSWindowShadowSecondaryWindow = 0x2,
};

@interface NSWindow ()
- (NSInspectorBar *)inspectorBar;
- (void)setInspectorBar:(NSInspectorBar *)bar;

@property (readonly) NSWindowShadowOptions shadowOptions;
@property CGFloat titlebarAlphaValue;

#if HAVE(NSSCROLLVIEW_SEPARATOR_TRACKING_ADAPTER)
- (BOOL)registerScrollViewSeparatorTrackingAdapter:(NSObject<NSScrollViewSeparatorTrackingAdapter> *)adapter;
- (void)unregisterScrollViewSeparatorTrackingAdapter:(NSObject<NSScrollViewSeparatorTrackingAdapter> *)adapter;
#endif

@end

@class LPLinkMetadata;

@interface NSPreviewRepresentingActivityItem ()
- (instancetype)initWithItem:(id)item linkMetadata:(LPLinkMetadata *)linkMetadata;
@end

#endif

@interface NSPopover (IPI)
@property (readonly) NSView *positioningView;
@end

@interface NSWorkspace (NSWorkspaceAccessibilityDisplayInternal_IPI)
+ (void)_invalidateAccessibilityDisplayValues;
@end

@interface NSInspectorBar (IPI)
- (void)_update;
@end

// FIXME: Move this above once <rdar://problem/70224980> is in an SDK.
@interface NSCursor ()
+ (void)hideUntilChanged;
@end

#if HAVE(NSWINDOW_SNAPSHOT_READINESS_HANDLER)
// FIXME: Move this above once <rdar://problem/112554759> is in an SDK.
@interface NSWindow (Staging_112554759)
typedef void (^NSWindowSnapshotReadinessHandler) (void);
- (NSWindowSnapshotReadinessHandler)_holdResizeSnapshotWithReason:(NSString *)reason;
@end
#endif
