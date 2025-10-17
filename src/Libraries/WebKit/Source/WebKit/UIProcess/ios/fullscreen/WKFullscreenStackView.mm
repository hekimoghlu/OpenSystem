/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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

#if ENABLE(FULLSCREEN_API) && PLATFORM(IOS_FAMILY)
#import "WKFullscreenStackView.h"

#import "UIKitSPI.h"
#import <UIKit/UIVisualEffectView.h>
#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RetainPtr.h>

SOFTLINK_AVKIT_FRAMEWORK()
SOFT_LINK_CLASS_OPTIONAL(AVKit, AVBackgroundView)

@interface WKFullscreenStackView () {

#if PLATFORM(APPLETV)
    RetainPtr<UIView> _backgroundView;
#else
    RetainPtr<AVBackgroundView> _backgroundView;
#endif
}
@end

@implementation WKFullscreenStackView

#pragma mark - External Interface

- (instancetype)init
{
    CGRect frame = CGRectMake(0, 0, 100, 100);
    self = [self initWithFrame:frame];

    if (!self)
        return nil;

    [self setClipsToBounds:YES];
#if !PLATFORM(APPLETV)
    _backgroundView = adoptNS([allocAVBackgroundViewInstance() initWithFrame:frame]);
    // FIXME: remove this once AVBackgroundView handles this. https://bugs.webkit.org/show_bug.cgi?id=188022
    [_backgroundView setClipsToBounds:YES];
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [_backgroundView.get().layer setContinuousCorners:YES];
ALLOW_DEPRECATED_DECLARATIONS_END
    [_backgroundView.get().layer setCornerRadius:16];
    [_backgroundView.get().layer setCornerCurve:kCACornerCurveCircular];
    [self addSubview:_backgroundView.get()];
#endif
    return self;
}

#if !PLATFORM(APPLETV)
- (void)addArrangedSubview:(UIView *)subview applyingMaterialStyle:(AVBackgroundViewMaterialStyle)materialStyle tintEffectStyle:(AVBackgroundViewTintEffectStyle)tintEffectStyle
{
    [_backgroundView.get() addSubview:subview applyingMaterialStyle:materialStyle tintEffectStyle:tintEffectStyle];
    [self addArrangedSubview:subview];
}
#endif

#pragma mark - UIView Overrides

- (void)layoutSubviews
{
    [_backgroundView.get() setFrame:self.bounds];
    [super layoutSubviews];
}

@end

#endif // ENABLE(FULLSCREEN_API) && PLATFORM(IOS_FAMILY)
