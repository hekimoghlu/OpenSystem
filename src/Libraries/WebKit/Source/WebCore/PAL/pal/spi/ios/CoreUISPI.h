/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <Foundation/Foundation.h>

#if USE(APPLE_INTERNAL_SDK)

#import <CoreUI/CUICatalog.h>
#import <CoreUI/CUIStyleEffectConfiguration.h>

#else

@interface CUIStyleEffectConfiguration : NSObject <NSCopying>
@end

@interface CUIStyleEffectConfiguration ()
@property (nonatomic) BOOL useSimplifiedEffect;
@property (nonatomic, copy) NSString *appearanceName;
@end

@interface CUICatalog : NSObject
@end

@interface CUICatalog ()
- (BOOL)drawGlyphs:(const CGGlyph[])glyphs atPositions:(const CGPoint[])positions inContext:(CGContextRef)context withFont:(CTFontRef)font count:(NSUInteger)count stylePresetName:(NSString *)stylePresetName styleConfiguration:(CUIStyleEffectConfiguration *)styleConfiguration foregroundColor:(CGColorRef)foregroundColor;
@end

#endif
