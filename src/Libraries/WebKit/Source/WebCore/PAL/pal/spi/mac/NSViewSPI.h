/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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

#if USE(APPKIT)

#import <pal/spi/cocoa/QuartzCoreSPI.h>

#if USE(APPLE_INTERNAL_SDK)
#import <AppKit/NSView_Private.h>
#else

#if USE(NSVIEW_SEMANTICCONTEXT)

typedef NS_ENUM(NSInteger, NSViewSemanticContext) {
    NSViewSemanticContextForm = 8,
};

#endif

@interface NSView ()

- (NSView *)_findLastViewInKeyViewLoop;

#if USE(NSVIEW_SEMANTICCONTEXT)
@property (nonatomic, setter=_setSemanticContext:) NSViewSemanticContext _semanticContext;
#endif

#if !HAVE(NSVIEW_CLIPSTOBOUNDS_API)
@property BOOL clipsToBounds;
#endif

@end

#endif // USE(APPLE_INTERNAL_SDK)

@interface NSView () <CALayerDelegate>
@end

@interface NSView (SubviewsIvar)
@property (assign, setter=_setSubviewsIvar:) NSMutableArray<__kindof NSView *> *_subviewsIvar;
@end

#endif // PLATFORM(MAC)
