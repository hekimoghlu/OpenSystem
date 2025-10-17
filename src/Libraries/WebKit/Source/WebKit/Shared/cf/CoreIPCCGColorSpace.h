/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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

#if PLATFORM(COCOA)

#import <CoreGraphics/CoreGraphics.h>
#import <WebCore/Color.h>
#import <WebCore/ColorSpace.h>
#import <WebCore/ColorSpaceCG.h>
#import <wtf/RetainPtr.h>

using CGColorSpaceSerialization = std::variant<WebCore::ColorSpace, RetainPtr<CFStringRef>, RetainPtr<CFTypeRef>>;

namespace WebKit {
class CoreIPCCGColorSpace {
public:
    CoreIPCCGColorSpace(CGColorSpaceRef cgColorSpace)
    {
        if (auto colorSpace = WebCore::colorSpaceForCGColorSpace(cgColorSpace))
            m_cgColorSpace = *colorSpace;
        else if (RetainPtr<CFStringRef> name = CGColorSpaceGetName(cgColorSpace))
            m_cgColorSpace = WTFMove(name);
        else if (auto propertyList = adoptCF(CGColorSpaceCopyPropertyList(cgColorSpace)))
            m_cgColorSpace = WTFMove(propertyList);
        else
            // FIXME: This should be removed once we can prove only non-null cgColorSpaces.
            m_cgColorSpace = WebCore::ColorSpace::SRGB;
    }

    CoreIPCCGColorSpace(CGColorSpaceSerialization data)
        : m_cgColorSpace(data)
    {
    }

    RetainPtr<CGColorSpaceRef> toCF() const
    {
        auto colorSpace = WTF::switchOn(m_cgColorSpace,
            [](WebCore::ColorSpace colorSpace) -> RetainPtr<CGColorSpaceRef> {
                return RetainPtr { cachedNullableCGColorSpace(colorSpace) };
            },
            [](RetainPtr<CFStringRef> name) -> RetainPtr<CGColorSpaceRef> {
                return adoptCF(CGColorSpaceCreateWithName(name.get()));
            },
            [](RetainPtr<CFTypeRef> propertyList) -> RetainPtr<CGColorSpaceRef> {
                return adoptCF(CGColorSpaceCreateWithPropertyList(propertyList.get()));
            }
        );
        if (UNLIKELY(!colorSpace))
            return nullptr;
        return colorSpace;
    }

    CGColorSpaceSerialization m_cgColorSpace;
};

}

#endif
