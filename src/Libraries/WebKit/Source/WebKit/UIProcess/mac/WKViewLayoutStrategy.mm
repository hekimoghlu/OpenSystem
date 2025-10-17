/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#import "WKViewLayoutStrategy.h"

#if PLATFORM(MAC)

#import "WebPageProxy.h"
#import "WebViewImpl.h"
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/MachSendRight.h>

@interface WKViewViewSizeLayoutStrategy : WKViewLayoutStrategy
@end

@interface WKViewFixedSizeLayoutStrategy : WKViewLayoutStrategy
@end

@interface WKViewDynamicSizeComputedFromViewScaleLayoutStrategy : WKViewLayoutStrategy
@end

@interface WKViewDynamicSizeComputedFromMinimumDocumentSizeLayoutStrategy : WKViewLayoutStrategy
@end

@implementation WKViewLayoutStrategy

+ (instancetype)layoutStrategyWithPage:(NakedRef<WebKit::WebPageProxy>)page view:(NSView *)view viewImpl:(NakedRef<WebKit::WebViewImpl>)webViewImpl mode:(WKLayoutMode)mode
{
    RetainPtr<WKViewLayoutStrategy> strategy;

    switch (mode) {
    case kWKLayoutModeFixedSize:
        strategy = adoptNS([[WKViewFixedSizeLayoutStrategy alloc] initWithPage:page.copyRef() view:view viewImpl:webViewImpl.copyRef() mode:mode]);
        break;
    case kWKLayoutModeDynamicSizeComputedFromViewScale:
        strategy = adoptNS([[WKViewDynamicSizeComputedFromViewScaleLayoutStrategy alloc] initWithPage:page.copyRef() view:view viewImpl:webViewImpl.copyRef() mode:mode]);
        break;
    case kWKLayoutModeDynamicSizeComputedFromMinimumDocumentSize:
        strategy = adoptNS([[WKViewDynamicSizeComputedFromMinimumDocumentSizeLayoutStrategy alloc] initWithPage:page.copyRef() view:view viewImpl:webViewImpl.copyRef() mode:mode]);
        break;
    case kWKLayoutModeViewSize:
    default:
        strategy = adoptNS([[WKViewViewSizeLayoutStrategy alloc] initWithPage:page.copyRef() view:view viewImpl:webViewImpl.copyRef() mode:mode]);
        break;
    }

    [strategy updateLayout];

    return strategy.autorelease();
}

- (instancetype)initWithPage:(NakedRef<WebKit::WebPageProxy>)page view:(NSView *)view viewImpl:(NakedRef<WebKit::WebViewImpl>)webViewImpl mode:(WKLayoutMode)mode
{
    self = [super init];

    if (!self)
        return nil;

    _page = page.ptr();
    _webViewImpl = webViewImpl.ptr();
    _view = view;
    _layoutMode = mode;

    return self;
}

- (void)invalidate
{
    _page = nullptr;
    _webViewImpl = nullptr;
    _view = nil;
}

- (WKLayoutMode)layoutMode
{
    return _layoutMode;
}

- (void)updateLayout
{
}

- (void)disableFrameSizeUpdates
{
}

- (void)enableFrameSizeUpdates
{
}

- (BOOL)frameSizeUpdatesDisabled
{
    return NO;
}

- (void)didChangeViewScale
{
}

- (void)willStartLiveResize
{
}

- (void)didEndLiveResize
{
}

- (void)didChangeFrameSize
{
    if (_webViewImpl->clipsToVisibleRect())
        _webViewImpl->updateViewExposedRect();
    _webViewImpl->setDrawingAreaSize(NSSizeToCGSize(_view.frame.size));
}

- (void)willChangeLayoutStrategy
{
}

@end

@implementation WKViewViewSizeLayoutStrategy

- (instancetype)initWithPage:(NakedRef<WebKit::WebPageProxy>)page view:(NSView *)view viewImpl:(NakedRef<WebKit::WebViewImpl>)webViewImpl mode:(WKLayoutMode)mode
{
    self = [super initWithPage:page.copyRef() view:view viewImpl:webViewImpl.copyRef() mode:mode];

    if (!self)
        return nil;

    page->setUseFixedLayout(false);

    return self;
}

- (void)updateLayout
{
}

@end

@implementation WKViewFixedSizeLayoutStrategy

- (instancetype)initWithPage:(NakedRef<WebKit::WebPageProxy>)page view:(NSView *)view viewImpl:(NakedRef<WebKit::WebViewImpl>)webViewImpl mode:(WKLayoutMode)mode
{
    self = [super initWithPage:page.copyRef() view:view viewImpl:webViewImpl.copyRef() mode:mode];

    if (!self)
        return nil;

    page->setUseFixedLayout(true);

    return self;
}

- (void)updateLayout
{
}

@end

@implementation WKViewDynamicSizeComputedFromViewScaleLayoutStrategy

- (instancetype)initWithPage:(NakedRef<WebKit::WebPageProxy>)page view:(NSView *)view viewImpl:(NakedRef<WebKit::WebViewImpl>)webViewImpl mode:(WKLayoutMode)mode
{
    self = [super initWithPage:page.copyRef() view:view viewImpl:webViewImpl.copyRef() mode:mode];

    if (!self)
        return nil;

    page->setUseFixedLayout(true);

    return self;
}

- (void)updateLayout
{
    CGFloat inverseScale = 1 / _page->viewScaleFactor();
    _webViewImpl->setFixedLayoutSize(CGSizeMake(_view.frame.size.width * inverseScale, _view.frame.size.height * inverseScale));
}

- (void)didChangeViewScale
{
    [super didChangeViewScale];

    [self updateLayout];
}

- (void)didChangeFrameSize
{
    [super didChangeFrameSize];

    if (self.frameSizeUpdatesDisabled)
        return;

    [self updateLayout];
}

@end

@implementation WKViewDynamicSizeComputedFromMinimumDocumentSizeLayoutStrategy

- (instancetype)initWithPage:(NakedRef<WebKit::WebPageProxy>)page view:(NSView *)view viewImpl:(NakedRef<WebKit::WebViewImpl>)webViewImpl mode:(WKLayoutMode)mode
{
    self = [super initWithPage:page.copyRef() view:view viewImpl:webViewImpl.copyRef() mode:mode];

    if (!self)
        return nil;

    _page->setShouldScaleViewToFitDocument(true);

    return self;
}

- (void)updateLayout
{
}

- (void)willChangeLayoutStrategy
{
    _page->setShouldScaleViewToFitDocument(false);
    _page->scaleView(1);
}

@end

#endif // PLATFORM(MAC)
