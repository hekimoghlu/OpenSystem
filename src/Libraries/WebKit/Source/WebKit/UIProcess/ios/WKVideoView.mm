/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#include "config.h"
#include "WKVideoView.h"

#if PLATFORM(IOS_FAMILY)

#include <wtf/RetainPtr.h>

@implementation WKVideoView {
    RetainPtr<WebAVPlayerLayerView> _playerView;
}

+ (Class)layerClass
{
    return [CALayer class];
}

- (id)initWithFrame:(CGRect)frame playerView:(WebAVPlayerLayerView *)playerView
{
    self = [super initWithFrame:frame];
    if (!self)
        return nil;

    _playerView = playerView;
    [self addSubview:playerView];

    return self;
}

- (CALayer *)playerLayer
{
    return _playerView.get().layer;
}

- (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event
{
    return [self pointInside:point withEvent:event] ? self : nil;
}

- (void)layoutSubviews
{
    for (UIView *subview in self.subviews)
        subview.frame = self.bounds;
}
@end
#endif
