/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#import <wtf/Platform.h>

#if PLATFORM(MAC)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSImage_Private.h>

#else

NS_ASSUME_NONNULL_BEGIN

@interface NSImage ()
+ (instancetype)imageWithImageRep:(NSImageRep *)imageRep;
- (void)lockFocusWithRect:(NSRect)rect context:(nullable NSGraphicsContext *)context hints:(nullable NSDictionary *)hints flipped:(BOOL)flipped;
@end

@interface NSImage (NSSystemSymbols)
+ (nullable NSImage *)_imageWithSystemSymbolName:(NSString *) symbolName;
@end

NS_ASSUME_NONNULL_END

#endif

NS_ASSUME_NONNULL_BEGIN

@interface _NSSVGImageRep : NSImageRep
- (nullable instancetype)initWithData:(NSData *)data;
@end

NS_ASSUME_NONNULL_END

#if HAVE(ALTERNATE_ICONS)

NS_ASSUME_NONNULL_BEGIN

extern const NSImageHintKey NSImageHintSymbolFont;
extern const NSImageHintKey NSImageHintSymbolScale;

NS_ASSUME_NONNULL_END

#endif // HAVE(ALTERNATE_ICONS)

#endif // USE(APPLE_INTERNAL_SDK)
