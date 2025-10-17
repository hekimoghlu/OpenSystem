/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#import "config.h"
#import "WKWebViewPrivateForTestingMac.h"

#if PLATFORM(MAC)

#import "AudioSessionRoutingArbitratorProxy.h"
#import "WKNSData.h"
#import "WKWebViewMac.h"
#import "WebColorPicker.h"
#import "WebPageProxy.h"
#import "WebProcessProxy.h"
#import "WebViewImpl.h"
#import "_WKFrameHandleInternal.h"
#import <WebCore/ColorCocoa.h>

@implementation WKWebView (WKTestingMac)

- (void)_requestControlledElementID
{
    if (_page)
        _page->requestControlledElementID();
}

- (void)_handleControlledElementIDResponse:(NSString *)identifier
{
    // Overridden by subclasses.
}

- (void)_handleAcceptedCandidate:(NSTextCheckingResult *)candidate
{
    _impl->handleAcceptedCandidate(candidate);
}

- (void)_didHandleAcceptedCandidate
{
    // Overridden by subclasses.
}

- (void)_didUpdateCandidateListVisibility:(BOOL)visible
{
    // Overridden by subclasses.
}

- (void)_forceRequestCandidates
{
    _impl->forceRequestCandidatesForTesting();
}

- (BOOL)_shouldRequestCandidates
{
    return _impl->shouldRequestCandidates();
}

- (BOOL)_allowsInlinePredictions
{
#if HAVE(INLINE_PREDICTIONS)
    return _impl->allowsInlinePredictions();
#else
    return NO;
#endif
}

- (void)_insertText:(id)string replacementRange:(NSRange)replacementRange
{
    [self insertText:string replacementRange:replacementRange];
}

- (NSRect)_candidateRect
{
    if (!_page->editorState().postLayoutData)
        return NSZeroRect;
    return _page->editorState().postLayoutData->selectionBoundingRect;
}

- (void)viewDidChangeEffectiveAppearance
{
    // This can be called during [super initWithCoder:] and [super initWithFrame:].
    // That is before _impl is ready to be used, so check. <rdar://problem/39611236>
    if (!_impl)
        return;

    _impl->effectiveAppearanceDidChange();
}

- (NSSet<NSView *> *)_pdfHUDs
{
    return _impl->pdfHUDs();
}

- (NSMenu *)_activeMenu
{
    if (NSMenu *contextMenu = _page->activeContextMenu())
        return contextMenu;
    if (NSMenu *domPasteMenu = _impl->domPasteMenu())
        return domPasteMenu;
    return nil;
}

- (void)_retrieveAccessibilityTreeData:(void (^)(NSData *, NSError *))completionHandler
{
    _page->getAccessibilityTreeData([completionHandler = makeBlockPtr(completionHandler)] (API::Data* data) {
        completionHandler(wrapper(data), nil);
    });
}

- (BOOL)_secureEventInputEnabledForTesting
{
    return _impl->inSecureInputState();
}

- (void)_setSelectedColorForColorPicker:(NSColor *)color
{
    _page->colorPickerClient().didChooseColor(WebCore::colorFromCocoaColor(color));
}

- (void)_createFlagsChangedEventMonitorForTesting
{
    _impl->createFlagsChangedEventMonitor();
}

- (BOOL)_hasFlagsChangedEventMonitorForTesting
{
    return _impl->hasFlagsChangedEventMonitor();
}

@end

#endif // PLATFORM(MAC)
