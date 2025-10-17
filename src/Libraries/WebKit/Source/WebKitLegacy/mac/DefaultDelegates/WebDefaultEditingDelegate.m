/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
#import <WebKitLegacy/WebDefaultEditingDelegate.h>

#import <WebKitLegacy/DOM.h>
#import <WebKitLegacy/WebEditingDelegate.h>
#import <WebKitLegacy/WebEditingDelegatePrivate.h>
#import <WebKitLegacy/WebView.h>

@implementation WebDefaultEditingDelegate

static WebDefaultEditingDelegate *sharedDelegate = nil;

+ (WebDefaultEditingDelegate *)sharedEditingDelegate
{
    if (!sharedDelegate) {
        sharedDelegate = [[WebDefaultEditingDelegate alloc] init];
    }
    return sharedDelegate;
}

- (BOOL)webView:(WebView *)webView shouldBeginEditingInDOMRange:(DOMRange *)range
{
    return YES;
}

- (BOOL)webView:(WebView *)webView shouldEndEditingInDOMRange:(DOMRange *)range
{
    return YES;
}

- (BOOL)webView:(WebView *)webView shouldInsertNode:(DOMNode *)node replacingDOMRange:(DOMRange *)range givenAction:(WebViewInsertAction)action
{
    return YES;
}

- (BOOL)webView:(WebView *)webView shouldInsertText:(NSString *)text replacingDOMRange:(DOMRange *)range givenAction:(WebViewInsertAction)action
{
    return YES;
}

- (BOOL)webView:(WebView *)webView shouldDeleteDOMRange:(DOMRange *)range
{
    return YES;
}

- (BOOL)webView:(WebView *)webView shouldChangeSelectedDOMRange:(DOMRange *)currentRange toDOMRange:(DOMRange *)proposedRange affinity:(NSSelectionAffinity)selectionAffinity stillSelecting:(BOOL)flag
{
    return YES;
}

- (BOOL)webView:(WebView *)webView shouldApplyStyle:(DOMCSSStyleDeclaration *)style toElementsInDOMRange:(DOMRange *)range
{
    return YES;
}

- (BOOL)webView:(WebView *)webView shouldMoveRangeAfterDelete:(DOMRange *)range replacingRange:(DOMRange *)rangeToBeReplaced
{
    return YES;
}

- (BOOL)webView:(WebView *)webView shouldChangeTypingStyle:(DOMCSSStyleDeclaration *)currentStyle toStyle:(DOMCSSStyleDeclaration *)proposedStyle
{
    return YES;
}

- (BOOL)webView:(WebView *)webView doCommandBySelector:(SEL)selector
{
    return NO;
}

#if !PLATFORM(IOS_FAMILY)
- (void)webView:(WebView *)webView didWriteSelectionToPasteboard:(NSPasteboard *)pasteboard
{
}
#else
- (NSArray *)supportedPasteboardTypesForCurrentSelection
{
    return nil;
}

- (DOMDocumentFragment *)documentFragmentForPasteboardItemAtIndex:(NSInteger)index
{
    return nil;
}
#endif

- (void)webViewDidBeginEditing:(NSNotification *)notification
{
}

- (void)webViewDidChange:(NSNotification *)notification
{
}

- (void)webViewDidEndEditing:(NSNotification *)notification
{
}

- (void)webViewDidChangeTypingStyle:(NSNotification *)notification
{
}

- (void)webViewDidChangeSelection:(NSNotification *)notification
{
}

- (NSUndoManager *)undoManagerForWebView:(WebView *)webView
{
    return nil;
}

@end
