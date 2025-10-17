/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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

#if PLATFORM(IOS_FAMILY)
#import "WebAutocorrectionData.h"

#import "UIKitSPI.h"
#import <UIKit/UIKit.h>
#import <WebCore/FloatRect.h>
#import <wtf/RetainPtr.h>
#import <wtf/Vector.h>

namespace WebKit {

WebAutocorrectionData::WebAutocorrectionData(Vector<WebCore::FloatRect>&& textRects, std::optional<String>&& fontName, double pointSize, double weight)
{
    this->textRects = WTFMove(textRects);
    if (fontName.has_value())
        this->font = [UIFont fontWithName:WTFMove(*fontName) size:pointSize];
    else
        this->font = [UIFont systemFontOfSize:pointSize weight:weight];
}

WebAutocorrectionData::WebAutocorrectionData(const Vector<WebCore::FloatRect>& textRects, const RetainPtr<UIFont>& font)
{
    this->textRects = textRects;
    this->font = font;
}

std::optional<String> WebAutocorrectionData::fontName() const
{
    if ([font isSystemFont])
        return std::nullopt;
    return { { [font fontName] } };
}

double WebAutocorrectionData::fontPointSize() const
{
    return [font pointSize];
}

double WebAutocorrectionData::fontWeight() const
{
    return [[[font fontDescriptor] objectForKey:UIFontWeightTrait] doubleValue];
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
