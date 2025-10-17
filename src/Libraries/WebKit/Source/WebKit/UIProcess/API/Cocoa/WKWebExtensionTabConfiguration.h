/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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
#import <Foundation/Foundation.h>
#import <WebKit/WKFoundation.h>

@protocol WKWebExtensionTab;
@protocol WKWebExtensionWindow;

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

/*!
 @abstract A ``WKWebExtensionTabConfiguration`` object encapsulates configuration options for a tab in an extension.
 @discussion This class holds various options that influence the behavior and initial state of a tab.
 The app retains the discretion to disregard any or all of these options, or even opt not to create a tab.
 */
WK_CLASS_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_SWIFT_UI_ACTOR NS_SWIFT_NAME(WKWebExtension.TabConfiguration)
@interface WKWebExtensionTabConfiguration : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/*!
 @abstract Indicates the window where the tab should be opened.
 @discussion If this property is `nil`, no window was specified.
 */
@property (nonatomic, nullable, readonly, strong) id <WKWebExtensionWindow> window;

/*! @abstract Indicates the position where the tab should be opened within the window. */
@property (nonatomic, readonly) NSUInteger index;

/*!
 @abstract Indicates the parent tab with which the tab should be related.
 @discussion If this property is `nil`, no parent tab was specified.
 */
@property (nonatomic, nullable, readonly, strong) id <WKWebExtensionTab> parentTab;

/*!
 @abstract Indicates the initial URL for the tab.
 @discussion If this property is `nil`, the app's default "start page" should appear in the tab.
 */
@property (nonatomic, nullable, readonly, copy) NSURL *url;

/*!
 @abstract Indicates whether the tab should be the active tab.
 @discussion If this property is `YES`, the tab should be made active in the window, ensuring it is
 the frontmost tab. Being active implies the tab is also selected. If this property is `NO`, the tab shouldn't
 affect the currently active tab.
 */
@property (nonatomic, readonly) BOOL shouldBeActive;

/*!
 @abstract Indicates whether the tab should be added to the current tab selection.
 @discussion If this property is `YES`, the tab should be part of the current selection, but not necessarily
 become the active tab unless ``shouldBeActive`` is also `YES`. If this property is `NO`, the tab shouldn't
 be part of the current selection.
 */
@property (nonatomic, readonly) BOOL shouldAddToSelection;

/*! @abstract Indicates whether the tab should be pinned. */
@property (nonatomic, readonly) BOOL shouldBePinned;

/*! @abstract Indicates whether the tab should be muted. */
@property (nonatomic, readonly) BOOL shouldBeMuted;

/*! @abstract Indicates whether reader mode in the tab should be active. */
@property (nonatomic, readonly) BOOL shouldReaderModeBeActive;

@end

WK_HEADER_AUDIT_END(nullability, sendability)
