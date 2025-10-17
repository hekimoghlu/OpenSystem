/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#import "_WKThumbnailViewInternal.h"

#if PLATFORM(MAC)

#import "ImageOptions.h"
#import "WKAPICast.h"
#import "WKView.h"
#import "WKViewInternal.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import <WebCore/ShareableBitmap.h>
#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <wtf/MathExtras.h>
#import <wtf/NakedPtr.h>
#import <wtf/SystemTracing.h>

// FIXME: Make it possible to leave a snapshot of the content presented in the WKView while the thumbnail is live.
// FIXME: Don't make new speculative tiles while thumbnailed.
// FIXME: Hide scrollbars in the thumbnail.
// FIXME: We should re-use existing tiles for unparented views, if we have them (we need to know if they've been purged; if so, repaint at scaled-down size).
// FIXME: We should switch to the low-resolution scale if a view we have high-resolution tiles for repaints.

@implementation _WKThumbnailView {
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    RetainPtr<WKView> _wkView;
ALLOW_DEPRECATED_DECLARATIONS_END
    RetainPtr<WKWebView> _wkWebView;
    NakedPtr<WebKit::WebPageProxy> _webPageProxy;

    BOOL _originalMayStartMediaWhenInWindow;
    BOOL _originalSourceViewIsInWindow;

    BOOL _snapshotWasDeferred;
    CGFloat _lastSnapshotScale;
    CGSize _lastSnapshotMaximumSize;

    RetainPtr<NSColor> _overrideBackgroundColor;
}

@synthesize _waitingForSnapshot;
@synthesize _sublayerVerticalTranslationAmount;

- (instancetype)initWithFrame:(NSRect)frame
{
    if (!(self = [super initWithFrame:frame]))
        return nil;

    self.wantsLayer = YES;
    _scale = 1;
    _lastSnapshotScale = NAN;
    
    return self;
}

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
- (instancetype)initWithFrame:(NSRect)frame fromWKView:(WKView *)wkView
{
    if (!(self = [self initWithFrame:frame]))
        return nil;

    _wkView = wkView;
    _webPageProxy = WebKit::toImpl([_wkView pageRef]);
    _originalMayStartMediaWhenInWindow = _webPageProxy->mayStartMediaWhenInWindow();
    _originalSourceViewIsInWindow = !![_wkView window];

    return self;
}
ALLOW_DEPRECATED_DECLARATIONS_END

- (instancetype)initWithFrame:(NSRect)frame fromWKWebView:(WKWebView *)webView
{
    if (!(self = [self initWithFrame:frame]))
        return nil;
    
    _wkWebView = webView;
    _webPageProxy = [_wkWebView _page];
    _originalMayStartMediaWhenInWindow = _webPageProxy->mayStartMediaWhenInWindow();
    _originalSourceViewIsInWindow = !![_wkWebView window];
    
    return self;
}

- (BOOL)isFlipped
{
    return YES;
}

- (BOOL)wantsUpdateLayer
{
    return YES;
}

- (void)updateLayer
{
    [super updateLayer];

    NSColor *backgroundColor = self.overrideBackgroundColor ?: [NSColor quaternaryLabelColor];
    self.layer.backgroundColor = backgroundColor.CGColor;
}

- (void)requestSnapshot
{
    if (_waitingForSnapshot) {
        _snapshotWasDeferred = YES;
        return;
    }

    tracePoint(TakeSnapshotStart);
    _waitingForSnapshot = YES;

    RetainPtr<_WKThumbnailView> thumbnailView = self;
    WebCore::IntRect snapshotRect(WebCore::IntPoint(), _webPageProxy->viewSize() - WebCore::IntSize(0, _webPageProxy->topContentInset()));
    WebKit::SnapshotOptions options { WebKit::SnapshotOption::InViewCoordinates, WebKit::SnapshotOption::UseScreenColorSpace };
    WebCore::IntSize bitmapSize = snapshotRect.size();
    bitmapSize.scale(_scale * _webPageProxy->deviceScaleFactor());

    if (!CGSizeEqualToSize(_maximumSnapshotSize, CGSizeZero)) {
        double sizeConstraintScale = 1;
        if (_maximumSnapshotSize.width)
            sizeConstraintScale = CGFloatMin(sizeConstraintScale, _maximumSnapshotSize.width / bitmapSize.width());
        if (_maximumSnapshotSize.height)
            sizeConstraintScale = CGFloatMin(sizeConstraintScale, _maximumSnapshotSize.height / bitmapSize.height());
        bitmapSize = WebCore::IntSize(CGCeiling(bitmapSize.width() * sizeConstraintScale), CGCeiling(bitmapSize.height() * sizeConstraintScale));
    }

    _lastSnapshotScale = _scale;
    _lastSnapshotMaximumSize = _maximumSnapshotSize;
    _webPageProxy->takeSnapshot(snapshotRect, bitmapSize, options, [thumbnailView](std::optional<WebCore::ShareableBitmap::Handle>&& imageHandle) {
        if (!imageHandle)
            return;
        auto bitmap = WebCore::ShareableBitmap::create(WTFMove(*imageHandle), WebCore::SharedMemory::Protection::ReadOnly);
        RetainPtr<CGImageRef> cgImage = bitmap ? bitmap->makeCGImage() : nullptr;
        tracePoint(TakeSnapshotEnd, !!cgImage);
        [thumbnailView _didTakeSnapshot:cgImage.get()];
    });
}

- (void)setOverrideBackgroundColor:(NSColor *)overrideBackgroundColor
{
    if ([_overrideBackgroundColor isEqual:overrideBackgroundColor])
        return;

    _overrideBackgroundColor = overrideBackgroundColor;
    [self setNeedsDisplay:YES];
}

- (NSColor *)overrideBackgroundColor
{
    return _overrideBackgroundColor.get();
}

- (void)_viewWasUnparented
{
    if (!_exclusivelyUsesSnapshot) {
        self._sublayerVerticalTranslationAmount = 0;
        if (_wkView) {
            [_wkView _setThumbnailView:nil];
            [_wkView _setIgnoresAllEvents:NO];
        } else {
            ASSERT(_wkWebView);
            [_wkWebView _setThumbnailView:nil];
            [_wkWebView _setIgnoresAllEvents:NO];
        }
        _webPageProxy->setMayStartMediaWhenInWindow(_originalMayStartMediaWhenInWindow);
    }

    if (_shouldKeepSnapshotWhenRemovedFromSuperview)
        return;

    self.layer.contents = nil;
    _lastSnapshotScale = NAN;
}

- (void)_viewWasParented
{
    if (_wkView && [_wkView _thumbnailView])
        return;
    if (_wkWebView && [_wkWebView _thumbnailView])
        return;

    if (!_exclusivelyUsesSnapshot && !_originalSourceViewIsInWindow)
        _webPageProxy->setMayStartMediaWhenInWindow(false);

    [self _requestSnapshotIfNeeded];

    if (!_exclusivelyUsesSnapshot) {
        self._sublayerVerticalTranslationAmount = -_webPageProxy->topContentInset();
        if (_wkView) {
            [_wkView _setThumbnailView:self];
            [_wkView _setIgnoresAllEvents:YES];
        } else {
            ASSERT(_wkWebView);
            [_wkWebView _setThumbnailView:self];
            [_wkWebView _setIgnoresAllEvents:YES];
        }
    }
}

- (void)_requestSnapshotIfNeeded
{
    if (self.layer.contents && _lastSnapshotScale == _scale && CGSizeEqualToSize(_lastSnapshotMaximumSize, _maximumSnapshotSize))
        return;

    [self requestSnapshot];
}

- (void)_didTakeSnapshot:(CGImageRef)image
{
    [self willChangeValueForKey:@"snapshotSize"];

    _snapshotSize = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
    _waitingForSnapshot = NO;
    self.layer.sublayers = @[];
    self.layer.contentsGravity = kCAGravityResizeAspectFill;
    self.layer.contents = (__bridge id)image;

    // If we got a scale change while snapshotting, we'll take another snapshot once the first one returns.
    if (_snapshotWasDeferred) {
        _snapshotWasDeferred = NO;
        [self _requestSnapshotIfNeeded];
    }

    [self didChangeValueForKey:@"snapshotSize"];
}

- (void)viewDidMoveToWindow
{
    if (self.window)
        [self _viewWasParented];
    else
        [self _viewWasUnparented];
}

- (void)setScale:(CGFloat)scale
{
    if (_scale == scale)
        return;

    _scale = scale;

    [self _requestSnapshotIfNeeded];

    auto scaleTransform = CATransform3DMakeScale(_scale, _scale, 1);
    self.layer.sublayerTransform = CATransform3DTranslate(scaleTransform, 0, _sublayerVerticalTranslationAmount, 0);
}

- (void)_setSublayerVerticalTranslationAmount:(CGFloat)amount
{
    if (WTF::areEssentiallyEqual(_sublayerVerticalTranslationAmount, amount))
        return;

    self.layer.sublayerTransform = CATransform3DTranslate(self.layer.sublayerTransform, 0, amount - _sublayerVerticalTranslationAmount, 0);
    _sublayerVerticalTranslationAmount = amount;
}

- (void)setMaximumSnapshotSize:(CGSize)maximumSnapshotSize
{
    if (CGSizeEqualToSize(_maximumSnapshotSize, maximumSnapshotSize))
        return;

    _maximumSnapshotSize = maximumSnapshotSize;

    [self _requestSnapshotIfNeeded];
}

- (void)_setThumbnailLayer:(CALayer *)layer
{
    self.layer.sublayers = layer ? @[ layer ] : @[ ];
}

- (CALayer *)_thumbnailLayer
{
    if (!self.layer.sublayers.count)
        return nil;

    return [self.layer.sublayers objectAtIndex:0];
}

@end

#endif // PLATFORM(MAC)
