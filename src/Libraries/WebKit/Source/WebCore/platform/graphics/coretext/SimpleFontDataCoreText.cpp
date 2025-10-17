/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#include "Font.h"

#include <CoreText/CoreText.h>
#include <pal/spi/cf/CoreTextSPI.h>

namespace WebCore {

static CTParagraphStyleRef paragraphStyleWithCompositionLanguageNone()
{
    static LazyNeverDestroyed<RetainPtr<CTParagraphStyleRef>> paragraphStyle;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&] {
        paragraphStyle.construct(CTParagraphStyleCreate(nullptr, 0));
        CTParagraphStyleSetCompositionLanguage(paragraphStyle.get().get(), kCTCompositionLanguageNone);
    });
    return paragraphStyle.get().get();
}

static CFNumberRef zeroValue()
{
    static LazyNeverDestroyed<RetainPtr<CFNumberRef>> zeroValue;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&] {
        const float zero = 0;
        zeroValue.construct(adoptCF(CFNumberCreate(kCFAllocatorDefault, kCFNumberFloatType, &zero)));
    });
    return zeroValue.get().get();
}

RetainPtr<CFDictionaryRef> Font::getCFStringAttributes(bool enableKerning, FontOrientation orientation, const AtomString& locale) const
{
    std::array<CFTypeRef, 5> keys;
    std::array<CFTypeRef, 5> values;

    keys[0] = kCTFontAttributeName;
    values[0] = platformData().ctFont();
    size_t count = 1;

    RetainPtr<CFStringRef> localeString;
    if (!locale.isEmpty()) {
        localeString = locale.string().createCFString();
        keys[count] = kCTLanguageAttributeName;
        values[count] = localeString.get();
        ++count;
    }

    keys[count] = kCTParagraphStyleAttributeName;
    values[count] = paragraphStyleWithCompositionLanguageNone();
    ++count;

    if (!enableKerning) {
        keys[count] = kCTKernAttributeName;
        values[count] = zeroValue();
        ++count;
    }

    if (orientation == FontOrientation::Vertical) {
        keys[count] = kCTVerticalFormsAttributeName;
        values[count] = kCFBooleanTrue;
        ++count;
    }

    ASSERT(count <= std::size(keys));

    return adoptCF(CFDictionaryCreate(kCFAllocatorDefault, keys.data(), values.data(), count, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
}

} // namespace WebCore
