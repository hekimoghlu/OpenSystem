/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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
#pragma once

#if ENABLE(ATTACHMENT_ELEMENT) && PLATFORM(COCOA)

#include "FloatRect.h"
#include "RenderAttachment.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS NSDictionary;

typedef const struct __CTFont* CTFontRef;
typedef const struct __CTLine * CTLineRef;

namespace WebCore {

class Image;

enum class AttachmentLayoutStyle : uint8_t { NonSelected, Selected };
struct AttachmentLayout {
    explicit AttachmentLayout(const RenderAttachment&, AttachmentLayoutStyle style = AttachmentLayoutStyle::NonSelected);
    
    float widthPadding { 0 };
    CGFloat wrappingWidth { 0 };
    FloatRect iconRect;
    FloatRect iconBackgroundRect;
    FloatRect attachmentRect;
    FloatRect progressRect;
    AttachmentLayoutStyle style;
    float progress { 0 };
    bool excludeTypographicLeading { false };
    RefPtr<Image> icon;
    RefPtr<Image> thumbnailIcon;
    Vector<CGPoint> origins;
    int baseline { 0 };
    bool hasProgress { false };
    
    struct LabelLine {
        FloatRect rect;
        FloatRect backgroundRect;
        RetainPtr<CTLineRef> line;
        RetainPtr<CTFontRef> font;
    };
    
    FloatRect subtitleTextRect;
    Vector<LabelLine> lines;
    
    CGFloat contentYOrigin { 0 };
    void layOutSubtitle(const RenderAttachment&);
    void layOutTitle(const RenderAttachment&);
    void buildWrappedLines(String&, CTFontRef, NSDictionary*, unsigned);
    void buildSingleLine(const String&, CTFontRef, NSDictionary*);
    void addLine(CTFontRef, CTLineRef, bool);
};

} // namespace WebCore

#endif // ENABLE(ATTACHMENT_ELEMENT) && PLATFORM(COCOA)
