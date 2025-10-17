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
#import <WebKit/WKFoundation.h>

#import <WebKit/WKWebProcessPlugInBrowserContextController.h>
#import <WebKit/WKWebProcessPlugInNodeHandle.h>
#import <WebKit/WKWebProcessPlugInRangeHandle.h>

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, WKEditorInsertAction) {
    WKEditorInsertActionTyped,
    WKEditorInsertActionPasted,
    WKEditorInsertActionDropped,
} WK_API_AVAILABLE(macos(10.12.4), ios(10.3));

WK_API_AVAILABLE(macos(10.12.4), ios(10.3))
@protocol WKWebProcessPlugInEditingDelegate <NSObject>

@optional

- (BOOL)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller shouldInsertText:(NSString *)text replacingRange:(WKWebProcessPlugInRangeHandle *)range givenAction:(WKEditorInsertAction)action;
#if TARGET_OS_IPHONE
- (BOOL)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller shouldChangeSelectedRange:(WKDOMRange *)currentRange toRange:(WKDOMRange *)proposedRange affinity:(UITextStorageDirection)selectionAffinity stillSelecting:(BOOL)stillSelecting WK_API_AVAILABLE(ios(11.0));
#else
- (BOOL)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller shouldChangeSelectedRange:(WKDOMRange *)currentRange toRange:(WKDOMRange *)proposedRange affinity:(NSSelectionAffinity)selectionAffinity stillSelecting:(BOOL)stillSelecting WK_API_AVAILABLE(macos(10.13));
#endif
- (void)_webProcessPlugInBrowserContextControllerDidChangeByEditing:(WKWebProcessPlugInBrowserContextController *)controller;
- (void)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller willWriteRangeToPasteboard:(WKWebProcessPlugInRangeHandle *)range;
- (NSDictionary<NSString *, NSData *> *)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller pasteboardDataForRange:(WKWebProcessPlugInRangeHandle *)range;
- (void)_webProcessPlugInBrowserContextControllerDidWriteToPasteboard:(WKWebProcessPlugInBrowserContextController *)controller;
- (BOOL)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller performTwoStepDrop:(WKWebProcessPlugInNodeHandle *)fragment atDestination:(WKWebProcessPlugInRangeHandle *)destination isMove:(BOOL)isMove WK_API_AVAILABLE(macos(10.13), ios(11.0));

@end

NS_ASSUME_NONNULL_END
