/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#import "WebViewVisualIdentificationOverlay.h"

#if PLATFORM(COCOA)

#import "Color.h"
#import "WebCoreCALayerExtras.h"
#import <CoreText/CoreText.h>
#import <wtf/WeakObjCPtr.h>

#if PLATFORM(IOS_FAMILY)
#import <pal/ios/UIKitSoftLink.h>
#endif

static void *boundsObservationContext = &boundsObservationContext;

@interface WebViewVisualIdentificationOverlay () <CALayerDelegate>
@end

const void* const webViewVisualIdentificationOverlayKey = &webViewVisualIdentificationOverlayKey;

@implementation WebViewVisualIdentificationOverlay {
    RetainPtr<PlatformView> _view;

    RetainPtr<CALayer> _layer;
    RetainPtr<NSString> _kind;
}

+ (BOOL)shouldIdentifyWebViews
{
    static std::optional<BOOL> shouldIdentifyWebViews;
    if (!shouldIdentifyWebViews)
        shouldIdentifyWebViews = [[NSUserDefaults standardUserDefaults] boolForKey:@"WebKitDebugIdentifyWebViews"];
    return *shouldIdentifyWebViews;
}

+ (void)installForWebViewIfNeeded:(PlatformView *)view kind:(NSString *)kind deprecated:(BOOL)isDeprecated
{
    if (![self shouldIdentifyWebViews])
        return;
    auto overlay = adoptNS([[WebViewVisualIdentificationOverlay alloc] initWithWebView:view kind:kind deprecated:isDeprecated]);
    objc_setAssociatedObject(self, webViewVisualIdentificationOverlayKey, overlay.get(), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

- (instancetype)initWithWebView:(PlatformView *)webView kind:(NSString *)kind deprecated:(BOOL)isDeprecated
{
    self = [super init];
    if (!self)
        return nil;

    _kind = kind;

#if USE(APPKIT)
    _view = adoptNS([[NSView alloc] initWithFrame:webView.bounds]);
    [_view setWantsLayer:YES];
    [_view setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];
#else
    _view = adoptNS([PAL::allocUIViewInstance() initWithFrame:webView.bounds]);
    [_view setUserInteractionEnabled:NO];
    [_view setAutoresizingMask:UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight];
#endif
    [webView addSubview:_view.get()];

#if PLATFORM(MACCATALYST)
    _kind = [_kind stringByAppendingString:@" (macCatalyst)"];
#endif

    _layer = adoptNS([[CATiledLayer alloc] init]);
    [_layer setName:@"WebViewVisualIdentificationOverlay"];
    [_layer setFrame:CGRectMake(0, 0, [_view bounds].size.width, [_view bounds].size.height)];
    auto viewColor = isDeprecated ? WebCore::Color::red.colorWithAlphaByte(50) : WebCore::Color::blue.colorWithAlphaByte(32);
    [_layer setBackgroundColor:cachedCGColor(viewColor).get()];
    [_layer setZPosition:999];
    [_layer setDelegate:self];
    [_layer web_disableAllActions];
    [[_view layer] addSublayer:_layer.get()];

    [[_view layer] addObserver:self forKeyPath:@"bounds" options:0 context:boundsObservationContext];

    return self;
}

- (void)dealloc
{
    [[_view layer] removeObserver:self forKeyPath:@"bounds" context:boundsObservationContext];

    [super dealloc];
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary<NSKeyValueChangeKey, id> *)change context:(void *)context
{
    UNUSED_PARAM(keyPath);
    UNUSED_PARAM(object);
    UNUSED_PARAM(change);
    if (context == boundsObservationContext) {
        [_layer setFrame:CGRectMake(0, 0, [_view bounds].size.width, [_view bounds].size.height)];
        [_layer setNeedsDisplay];
    }
}

static RetainPtr<CTFontRef> createIdentificationFont()
{
    auto matrix = CGAffineTransformIdentity;
    return adoptCF(CTFontCreateWithName(CFSTR("Helvetica"), 20, &matrix));
}

constexpr CGFloat horizontalMargin = 15;
constexpr CGFloat verticalMargin = 5;

static void drawPattern(void *overlayPtr, CGContextRef ctx)
{
    WebViewVisualIdentificationOverlay *overlay = (WebViewVisualIdentificationOverlay *)overlayPtr;

    auto attributes = @{
        (id)kCTFontAttributeName : (id)createIdentificationFont().get(),
        (id)kCTForegroundColorFromContextAttributeName : @YES
    };
    auto attributedString = adoptCF(CFAttributedStringCreate(kCFAllocatorDefault, (__bridge CFStringRef)overlay->_kind.get(), (__bridge CFDictionaryRef)attributes));
    auto line = adoptCF(CTLineCreateWithAttributedString(attributedString.get()));

    CGSize textSize = [overlay->_kind sizeWithAttributes:attributes];

#if PLATFORM(IOS_FAMILY)
    CGContextScaleCTM(ctx, 1, -1);
    CGContextTranslateCTM(ctx, 0, -(textSize.height + verticalMargin) * 2);
#endif

    CGContextSetTextDrawingMode(ctx, kCGTextFill);

    CGContextSetTextPosition(ctx, 0, 0);
    CGContextSetFillColorWithColor(ctx, cachedCGColor(WebCore::Color::black).get());
    CTLineDraw(line.get(), ctx);

    CGContextSetTextPosition(ctx, 0, textSize.height + 5);
    CGContextSetFillColorWithColor(ctx, cachedCGColor(WebCore::Color::white).get());
    CTLineDraw(line.get(), ctx);
}

- (void)drawLayer:(CALayer *)layer inContext:(CGContextRef)ctx
{
    CGPatternCallbacks callbacks = { 0, &drawPattern, nullptr };
    auto patternSpace = adoptCF(CGColorSpaceCreatePattern(nullptr));
    CGContextSetFillColorSpace(ctx, patternSpace.get());

    CGSize textSize = [_kind sizeWithAttributes:@{ (id)kCTFontAttributeName : (id)createIdentificationFont().get() }];
    CGSize patternSize = CGSizeMake(textSize.width + horizontalMargin, (textSize.height + verticalMargin) * 2);
    auto pattern = adoptCF(CGPatternCreate(self, layer.bounds, CGAffineTransformMakeRotation(M_PI_4), patternSize.width, patternSize.height, kCGPatternTilingNoDistortion, true, &callbacks));
    CGFloat alpha = 0.5;
    CGContextSetFillPattern(ctx, pattern.get(), &alpha);

    CGContextFillRect(ctx, layer.bounds);
}

@end

#endif // PLATFORM(COCOA)
