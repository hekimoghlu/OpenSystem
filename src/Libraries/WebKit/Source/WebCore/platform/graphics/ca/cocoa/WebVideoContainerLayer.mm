/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
#import "WebVideoContainerLayer.h"

@implementation WebVideoContainerLayer
- (void)setPlayerLayer:(AVPlayerLayer *)playerLayer
{
    _playerLayer = playerLayer;
}

- (AVPlayerLayer *)playerLayer
{
    return _playerLayer.get();
}

- (void)setBounds:(CGRect)bounds
{
    [super setBounds:bounds];
    for (CALayer* layer in self.sublayers)
        layer.frame = bounds;
}

- (void)setPosition:(CGPoint)position
{
    if (!CATransform3DIsIdentity(self.transform)) {
        // Pre-apply the transform added in the WebProcess to fix <rdar://problem/18316542> to the position.
        position = CGPointApplyAffineTransform(position, CATransform3DGetAffineTransform(self.transform));
    }
    [super setPosition:position];
}

@end

