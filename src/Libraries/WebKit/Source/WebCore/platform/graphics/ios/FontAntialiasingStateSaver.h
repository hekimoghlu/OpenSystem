/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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
#ifndef FontAntialiasingStateSaver_h
#define FontAntialiasingStateSaver_h

#if PLATFORM(IOS_FAMILY)

#import <pal/spi/cg/CoreGraphicsSPI.h>

namespace WebCore {

class FontAntialiasingStateSaver {
    WTF_MAKE_NONCOPYABLE(FontAntialiasingStateSaver);
public:
    FontAntialiasingStateSaver(CGContextRef context, bool useOrientationDependentFontAntialiasing)
#if !PLATFORM(IOS_FAMILY_SIMULATOR)
        : m_context(context)
        , m_useOrientationDependentFontAntialiasing(useOrientationDependentFontAntialiasing)
#endif
    {
#if PLATFORM(IOS_FAMILY_SIMULATOR)
        UNUSED_PARAM(context);
        UNUSED_PARAM(useOrientationDependentFontAntialiasing);
#endif
    }

    ~FontAntialiasingStateSaver()
    {
#if !PLATFORM(IOS_FAMILY_SIMULATOR)
        if (m_useOrientationDependentFontAntialiasing)
            CGContextSetFontAntialiasingStyle(m_context, m_oldAntialiasingStyle);
#endif
    }

    void setup(bool isLandscapeOrientation)
    {
#if !PLATFORM(IOS_FAMILY_SIMULATOR)
    m_oldAntialiasingStyle = CGContextGetFontAntialiasingStyle(m_context);

    if (m_useOrientationDependentFontAntialiasing)
        CGContextSetFontAntialiasingStyle(m_context, isLandscapeOrientation ? kCGFontAntialiasingStyleFilterLight : kCGFontAntialiasingStyleUnfiltered);
#else
    UNUSED_PARAM(isLandscapeOrientation);
#endif
    }

private:
#if !PLATFORM(IOS_FAMILY_SIMULATOR)
    CGContextRef m_context;
    bool m_useOrientationDependentFontAntialiasing;
    CGFontAntialiasingStyle m_oldAntialiasingStyle;
#endif
};

}

#endif

#endif // FontAntialiasingStateSaver_h
