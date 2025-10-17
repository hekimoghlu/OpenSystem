/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#if !PLATFORM(IOS_FAMILY)

#import "WebContextMenuClient.h"

#import "WebDelegateImplementationCaching.h"
#import "WebElementDictionary.h"
#import "WebFrameInternal.h"
#import "WebFrameView.h"
#import "WebHTMLViewInternal.h"
#import "WebKitVersionChecks.h"
#import "WebNSPasteboardExtras.h"
#import "WebSharingServicePickerController.h"
#import "WebUIDelegatePrivate.h"
#import "WebViewInternal.h"
#import <WebCore/BitmapImage.h>
#import <WebCore/ContextMenu.h>
#import <WebCore/ContextMenuController.h>
#import <WebCore/DestinationColorSpace.h>
#import <WebCore/Document.h>
#import <WebCore/GraphicsContext.h>
#import <WebCore/ImageBuffer.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/LocalizedStrings.h>
#import <WebCore/Page.h>
#import <WebCore/RenderBoxInlines.h>
#import <WebCore/SharedBuffer.h>
#import <WebCore/SimpleRange.h>
#import <WebKitLegacy/DOMPrivate.h>
#import <pal/spi/mac/NSSharingServicePickerSPI.h>
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/URL.h>

using namespace WebCore;

@interface NSApplication ()
- (BOOL)isSpeaking;
- (void)speakString:(NSString *)string;
- (void)stopSpeaking:(id)sender;
@end

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebContextMenuClient);

WebContextMenuClient::WebContextMenuClient(WebView *webView)
#if ENABLE(SERVICE_CONTROLS)
    : WebSharingServicePickerClient(webView)
#else
    : m_webView(webView)
#endif
{
}

WebContextMenuClient::~WebContextMenuClient()
{
#if ENABLE(SERVICE_CONTROLS)
    if (m_sharingServicePickerController)
        [m_sharingServicePickerController clear];
#endif
}

void WebContextMenuClient::downloadURL(const URL& url)
{
    [m_webView _downloadURL:url];
}

void WebContextMenuClient::searchWithGoogle(const LocalFrame*)
{
    [m_webView _searchWithGoogleFromMenu:nil];
}

void WebContextMenuClient::lookUpInDictionary(LocalFrame* frame)
{
    WebHTMLView* htmlView = (WebHTMLView*)[[kit(frame) frameView] documentView];
    if(![htmlView isKindOfClass:[WebHTMLView class]])
        return;
    [htmlView _lookUpInDictionaryFromMenu:nil];
}

bool WebContextMenuClient::isSpeaking() const
{
    return [NSApp isSpeaking];
}

void WebContextMenuClient::speak(const String& string)
{
    [NSApp speakString:(NSString *)string];
}

void WebContextMenuClient::stopSpeaking()
{
    [NSApp stopSpeaking:nil];
}

bool WebContextMenuClient::clientFloatRectForNode(Node& node, FloatRect& rect) const
{
    RenderObject* renderer = node.renderer();
    if (!renderer) {
        // This method shouldn't be called in cases where the controlled node hasn't rendered.
        ASSERT_NOT_REACHED();
        return false;
    }

    if (!is<RenderBox>(*renderer))
        return false;
    auto& renderBox = downcast<RenderBox>(*renderer);

    LayoutRect layoutRect = renderBox.clientBoxRect();
    FloatQuad floatQuad = renderBox.localToAbsoluteQuad(FloatQuad(layoutRect));
    rect = floatQuad.boundingBox();

    return true;
}

#if HAVE(TRANSLATION_UI_SERVICES)

void WebContextMenuClient::handleTranslation(const TranslationContextMenuInfo& info)
{
    [m_webView _handleContextMenuTranslation:info];
}

#endif

#if ENABLE(SERVICE_CONTROLS)

void WebContextMenuClient::sharingServicePickerWillBeDestroyed(WebSharingServicePickerController &)
{
    m_sharingServicePickerController = nil;
}

WebCore::FloatRect WebContextMenuClient::screenRectForCurrentSharingServicePickerItem(WebSharingServicePickerController &)
{
    Page* page = [m_webView page];
    if (!page)
        return NSZeroRect;

    Node* node = page->contextMenuController().context().hitTestResult().innerNode();
    if (!node)
        return NSZeroRect;

    auto* frameView = node->document().view();
    if (!frameView) {
        // This method shouldn't be called in cases where the controlled node isn't in a rendered view.
        ASSERT_NOT_REACHED();
        return NSZeroRect;
    }

    FloatRect rect;
    if (!clientFloatRectForNode(*node, rect))
        return NSZeroRect;

    // FIXME: https://webkit.org/b/132915
    // Ideally we'd like to convert the content rect to screen coordinates without the lossy float -> int conversion.
    // Creating a rounded int rect works well in practice, but might still lead to off-by-one-pixel problems in edge cases.
    IntRect intRect = roundedIntRect(rect);
    return frameView->contentsToScreen(intRect);
}

RetainPtr<NSImage> WebContextMenuClient::imageForCurrentSharingServicePickerItem(WebSharingServicePickerController &)
{
    auto page = [m_webView page];
    if (!page)
        return nil;

    RefPtr node = page->contextMenuController().context().hitTestResult().innerNode();
    if (!node)
        return nil;

    RefPtr frameView = node->document().view();
    if (!frameView) {
        // This method shouldn't be called in cases where the controlled node isn't in a rendered view.
        ASSERT_NOT_REACHED();
        return nil;
    }

    FloatRect rect;
    if (!clientFloatRectForNode(*node, rect))
        return nil;

    // This is effectively a snapshot, and will be painted in an unaccelerated fashion in line with FrameSnapshotting.
    auto buffer = ImageBuffer::create(rect.size(), RenderingMode::Unaccelerated, RenderingPurpose::Unspecified, 1, DestinationColorSpace::SRGB(), ImageBufferPixelFormat::BGRA8);
    if (!buffer)
        return nil;

    Ref localFrame = frameView->frame();

    auto oldSelection = localFrame->selection().selection();
    localFrame->selection().setSelection(*makeRangeSelectingNode(*node), FrameSelection::SetSelectionOption::DoNotSetFocus);

    auto oldPaintBehavior = frameView->paintBehavior();
    frameView->setPaintBehavior(PaintBehavior::SelectionOnly);

    buffer->context().translate(-toFloatSize(rect.location()));
    frameView->paintContents(buffer->context(), roundedIntRect(rect));

    localFrame->selection().setSelection(oldSelection);
    frameView->setPaintBehavior(oldPaintBehavior);

    auto image = BitmapImage::create(ImageBuffer::sinkIntoNativeImage(WTFMove(buffer)));
    if (!image)
        return nil;

    return image->adapter().snapshotNSImage();
}

#endif

NSMenu *WebContextMenuClient::contextMenuForEvent(NSEvent *event, NSView *view, bool& isServicesMenu)
{
    isServicesMenu = false;

    Page* page = [m_webView page];
    if (!page)
        return nil;

#if ENABLE(SERVICE_CONTROLS)
    if (Image* image = page->contextMenuController().context().controlledImage()) {
        ASSERT(page->contextMenuController().context().hitTestResult().innerNode());

        RetainPtr<NSItemProvider> itemProvider = adoptNS([[NSItemProvider alloc] initWithItem:image->adapter().snapshotNSImage().get() typeIdentifier:@"public.image"]);

        bool isContentEditable = page->contextMenuController().context().hitTestResult().innerNode()->isContentEditable();
        m_sharingServicePickerController = adoptNS([[WebSharingServicePickerController alloc] initWithItems:@[ itemProvider.get() ] includeEditorServices:isContentEditable client:this style:NSSharingServicePickerStyleRollover]);

        isServicesMenu = true;
        return [m_sharingServicePickerController menu];
    }
#endif

    return [view menuForEvent:event];
}

void WebContextMenuClient::showContextMenu()
{
    auto page = [m_webView page];
    if (!page)
        return;
    auto* frame = page->contextMenuController().hitTestResult().innerNodeFrame();
    if (!frame)
        return;
    auto* frameView = frame->view();
    if (!frameView)
        return;

    NSView* view = frameView->documentView();
    IntPoint point = frameView->contentsToWindow(page->contextMenuController().hitTestResult().roundedPointInInnerNodeFrame());
    NSEvent* event = [NSEvent mouseEventWithType:NSEventTypeRightMouseDown location:point modifierFlags:0 timestamp:0 windowNumber:[[view window] windowNumber] context:0 eventNumber:0 clickCount:1 pressure:1];

    // Show the contextual menu for this event.
    bool isServicesMenu;
    if (NSMenu *menu = contextMenuForEvent(event, view, isServicesMenu)) {
        if (isServicesMenu)
            [menu popUpMenuPositioningItem:nil atLocation:[view convertPoint:point toView:nil] inView:view];
        else
            [NSMenu popUpContextMenu:menu withEvent:event forView:view];
    }
}

#endif // !PLATFORM(IOS_FAMILY)
