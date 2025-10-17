/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#import "WKUSDPreviewView.h"

#if USE(SYSTEM_PREVIEW)

#import "APIFindClient.h"
#import "APIUIClient.h"
#import "WKWebViewIOS.h"
#import "WebPageProxy.h"
#import <MobileCoreServices/MobileCoreServices.h>
#import <WebCore/FloatRect.h>
#import <WebCore/LocalizedStrings.h>
#import <WebCore/MIMETypeRegistry.h>
#import <WebCore/UTIUtilities.h>
#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <pal/spi/ios/SystemPreviewSPI.h>
#import <wtf/RetainPtr.h>
#import <wtf/Vector.h>

#import <pal/ios/QuickLookSoftLink.h>

SOFT_LINK_PRIVATE_FRAMEWORK(AssetViewer);
SOFT_LINK_CLASS(AssetViewer, ASVThumbnailView);

static NSString *getUTIForUSDMIMEType(const String& mimeType)
{
    if (!WebCore::MIMETypeRegistry::isUSDMIMEType(mimeType))
        return nil;

    return WebCore::UTIFromMIMEType(mimeType);
}

@interface WKUSDPreviewView () <ASVThumbnailViewDelegate>
@end

@implementation WKUSDPreviewView {
    RetainPtr<NSItemProvider> _itemProvider;
    RetainPtr<NSData> _data;
    RetainPtr<NSString> _suggestedFilename;
    RetainPtr<NSString> _mimeType;
    RetainPtr<QLItem> _item;
    RetainPtr<ASVThumbnailView> _thumbnailView;
    WKWebView *_webView;
}

- (instancetype)web_initWithFrame:(CGRect)frame webView:(WKWebView *)webView mimeType:(NSString *)mimeType
{
    if (!(self = [super initWithFrame:frame webView:webView]))
        return nil;

    UIColor *backgroundColor = [UIColor colorWithRed:(38. / 255) green:(38. / 255) blue:(38. / 255) alpha:1];
    self.backgroundColor = backgroundColor;

    _webView = webView;
    _mimeType = mimeType;

    UIScrollView *scrollView = webView.scrollView;
    [scrollView setMinimumZoomScale:1];
    [scrollView setMaximumZoomScale:1];
    [scrollView setBackgroundColor:backgroundColor];

    return self;
}

- (void)web_setContentProviderData:(NSData *)data suggestedFilename:(NSString *)filename
{
    _suggestedFilename = adoptNS([filename copy]);
    _data = adoptNS([data copy]);

    NSString *contentType = getUTIForUSDMIMEType(_mimeType.get());

    _item = adoptNS([PAL::allocQLItemInstance() initWithDataProvider:self contentType:contentType previewTitle:_suggestedFilename.get()]);
    [_item setUseLoadingTimeout:NO];

    _thumbnailView = adoptNS([allocASVThumbnailViewInstance() init]);
    [_thumbnailView setDelegate:self];
    [self setAutoresizingMask:UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight];

    [self setAutoresizesSubviews:YES];
    [self setClipsToBounds:YES];
    [self addSubview:_thumbnailView.get()];
    [self _layoutThumbnailView];

    auto screenBounds = UIScreen.mainScreen.bounds;
    CGFloat maxDimension = CGFloatMin(screenBounds.size.width, screenBounds.size.height);
    [_thumbnailView setMaxThumbnailSize:CGSizeMake(maxDimension, maxDimension)];

    [_thumbnailView setThumbnailItem:_item.get()];
}

- (void)_layoutThumbnailView
{
    if (_thumbnailView) {
        UIEdgeInsets safeAreaInsets = _webView._computedUnobscuredSafeAreaInset;
        UIEdgeInsets obscuredAreaInsets = _webView._computedObscuredInset;

        CGRect layoutFrame = UIEdgeInsetsInsetRect(self.frame, safeAreaInsets);

        layoutFrame.size.width -= obscuredAreaInsets.left + obscuredAreaInsets.right;
        layoutFrame.size.height -= obscuredAreaInsets.top + obscuredAreaInsets.bottom;

        [_thumbnailView setFrame:layoutFrame];
        [_thumbnailView setAutoresizingMask:UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight];
    }
}

#pragma mark ASVThumbnailViewDelegate

- (void)thumbnailView:(ASVThumbnailView *)thumbnailView wantsToPresentPreviewController:(QLPreviewController *)previewController forItem:(QLItem *)item
{
    RefPtr<WebKit::WebPageProxy> page = _webView->_page;

    // FIXME: When in element fullscreen, UIClient::presentingViewController() may not return the
    // WKFullScreenViewController even though that is the presenting view controller of the WKWebView.
    // We should call PageClientImpl::presentingViewController() instead.
    UIViewController *presentingViewController = page->uiClient().presentingViewController();

    [presentingViewController presentViewController:previewController animated:YES completion:nil];
}

#pragma mark WKWebViewContentProvider protocol

- (UIView *)web_contentView
{
    return self;
}

+ (BOOL)web_requiresCustomSnapshotting
{
    return false;
}

- (void)web_setMinimumSize:(CGSize)size
{
}

- (void)web_setOverlaidAccessoryViewsInset:(CGSize)inset
{
}

- (void)web_computedContentInsetDidChange
{
    [self _layoutThumbnailView];
}

- (void)web_setFixedOverlayView:(UIView *)fixedOverlayView
{
}

- (void)web_didSameDocumentNavigation:(WKSameDocumentNavigationType)navigationType
{
}

- (BOOL)web_isBackground
{
    return self.isBackground;
}

#pragma mark Find-in-Page

- (void)web_countStringMatches:(NSString *)string options:(_WKFindOptions)options maxCount:(NSUInteger)maxCount
{
    RefPtr<WebKit::WebPageProxy> page = _webView->_page;
    page->findClient().didCountStringMatches(page.get(), string, 0);
}

- (void)web_findString:(NSString *)string options:(_WKFindOptions)options maxCount:(NSUInteger)maxCount
{
    RefPtr<WebKit::WebPageProxy> page = _webView->_page;
    page->findClient().didFailToFindString(page.get(), string);
}

- (void)web_hideFindUI
{
}

#pragma mark QLPreviewItemDataProvider

- (NSData *)provideDataForItem:(QLItem *)item
{
    return _data.get();
}

@end

#endif
