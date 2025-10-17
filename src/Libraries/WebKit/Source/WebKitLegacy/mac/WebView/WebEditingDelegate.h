/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#import <WebKitLegacy/WebKitAvailability.h>

#if !TARGET_OS_IPHONE
#import <AppKit/AppKit.h>
#else
#import <WebKitLegacy/WAKAppKitStubs.h>
#endif

@class DOMCSSStyleDeclaration;
@class DOMNode;
@class DOMRange;
@class WebView;

typedef NS_ENUM(NSInteger, WebViewInsertAction) {
    WebViewInsertActionTyped,
    WebViewInsertActionPasted,
    WebViewInsertActionDropped,
} WEBKIT_ENUM_DEPRECATED_MAC(10_3, 10_14);

WEBKIT_DEPRECATED_MAC(10_3, 10_14)
@protocol WebEditingDelegate <NSObject>

@optional

- (BOOL)webView:(WebView *)webView shouldBeginEditingInDOMRange:(DOMRange *)range;
- (BOOL)webView:(WebView *)webView shouldEndEditingInDOMRange:(DOMRange *)range;
- (BOOL)webView:(WebView *)webView shouldInsertNode:(DOMNode *)node replacingDOMRange:(DOMRange *)range givenAction:(WebViewInsertAction)action;
- (BOOL)webView:(WebView *)webView shouldInsertText:(NSString *)text replacingDOMRange:(DOMRange *)range givenAction:(WebViewInsertAction)action;
- (BOOL)webView:(WebView *)webView shouldDeleteDOMRange:(DOMRange *)range;
- (BOOL)webView:(WebView *)webView shouldChangeSelectedDOMRange:(DOMRange *)currentRange toDOMRange:(DOMRange *)proposedRange affinity:(NSSelectionAffinity)selectionAffinity stillSelecting:(BOOL)flag;
- (BOOL)webView:(WebView *)webView shouldApplyStyle:(DOMCSSStyleDeclaration *)style toElementsInDOMRange:(DOMRange *)range;
- (BOOL)webView:(WebView *)webView shouldChangeTypingStyle:(DOMCSSStyleDeclaration *)currentStyle toStyle:(DOMCSSStyleDeclaration *)proposedStyle;
- (BOOL)webView:(WebView *)webView doCommandBySelector:(SEL)selector;
- (void)webViewDidBeginEditing:(NSNotification *)notification;
- (void)webViewDidChange:(NSNotification *)notification;
- (void)webViewDidEndEditing:(NSNotification *)notification;
- (void)webViewDidChangeTypingStyle:(NSNotification *)notification;
- (void)webViewDidChangeSelection:(NSNotification *)notification;
- (NSUndoManager *)undoManagerForWebView:(WebView *)webView;

@end
