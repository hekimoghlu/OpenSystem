/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#import "WebSystemBackdropLayer.h"

#import "GraphicsContextCG.h"
#import <QuartzCore/QuartzCore.h>

// FIXME: https://bugs.webkit.org/show_bug.cgi?id=146250
// These should provide the system recipes for the layers
// with the appropriate tinting, blending and blurring.

@implementation WebSystemBackdropLayer
@end

@implementation WebLightSystemBackdropLayer

- (id)init
{
    self = [super init];
    if (!self)
        return nil;

#ifndef NDEBUG
    [self setName:@"WebLightSystemBackdropLayer"];
#endif

    CGFloat components[4] = { 0.8, 0.8, 0.8, 0.8 };
    [super setBackgroundColor:adoptCF(CGColorCreate(WebCore::sRGBColorSpaceRef(), components)).get()];

    return self;
}

- (void)setBackgroundColor:(CGColorRef)backgroundColor
{
    // Empty implementation to stop clients pushing the wrong color.
    UNUSED_PARAM(backgroundColor);
}

@end

@implementation WebDarkSystemBackdropLayer

- (id)init
{
    self = [super init];
    if (!self)
        return nil;

#ifndef NDEBUG
    [self setName:@"WebDarkSystemBackdropLayer"];
#endif

    CGFloat components[4] = { 0.2, 0.2, 0.2, 0.8 };
    [super setBackgroundColor:adoptCF(CGColorCreate(WebCore::sRGBColorSpaceRef(), components)).get()];

    return self;
}

- (void)setBackgroundColor:(CGColorRef)backgroundColor
{
    // Empty implementation to stop clients pushing the wrong color.
    UNUSED_PARAM(backgroundColor);
}

@end
