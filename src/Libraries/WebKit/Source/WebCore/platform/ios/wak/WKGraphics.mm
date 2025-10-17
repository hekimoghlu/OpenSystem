/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#import "WKGraphics.h"

#if PLATFORM(IOS_FAMILY)

#import "WebCoreThreadInternal.h"
#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <wtf/StdLibExtras.h>

static inline void _FillRectsUsingOperation(CGContextRef context, std::span<const CGRect> rects, CGCompositeOperation op)
{
    auto integralRects = unsafeMakeSpan(reinterpret_cast<CGRect *>(alloca(sizeof(CGRect) * rects.size())), rects.size());
    
    assert(integralRects.data());
    
    for (size_t i = 0; i < rects.size(); ++i) {
        integralRects[i] = CGRectApplyAffineTransform (rects[i], CGContextGetCTM(context));
        integralRects[i] = CGRectIntegral (integralRects[i]);
        integralRects[i] = CGRectApplyAffineTransform (integralRects[i], CGAffineTransformInvert(CGContextGetCTM(context)));
    }
    CGCompositeOperation oldOp = CGContextGetCompositeOperation(context);
    CGContextSetCompositeOperation(context, op);
    CGContextFillRects(context, integralRects.data(), rects.size());
    CGContextSetCompositeOperation(context, oldOp);
}

void WKRectFill(CGContextRef context, CGRect aRect)
{
    if (aRect.size.width > 0 && aRect.size.height > 0) {
        CGContextSaveGState(context);
        if (aRect.size.width > 0 && aRect.size.height > 0)
            _FillRectsUsingOperation(context, singleElementSpan(aRect), kCGCompositeCopy);
        CGContextRestoreGState(context);
    }
}

void WKSetCurrentGraphicsContext(CGContextRef context)
{
    WebThreadContext* threadContext =  WebThreadCurrentContext();
    threadContext->currentCGContext = context;
}

CGContextRef WKGetCurrentGraphicsContext(void)
{
    WebThreadContext* threadContext =  WebThreadCurrentContext();
    return threadContext->currentCGContext;
}

#endif // PLATFORM(IOS_FAMILY)
