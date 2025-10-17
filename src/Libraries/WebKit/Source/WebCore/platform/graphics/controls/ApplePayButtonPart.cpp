/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
#include "ApplePayButtonPart.h"

#if ENABLE(APPLE_PAY)

#include "ControlFactory.h"

namespace WebCore {

Ref<ApplePayButtonPart> ApplePayButtonPart::create()
{
    return adoptRef(*new ApplePayButtonPart(ApplePayButtonType::Plain, ApplePayButtonStyle::White, { }));
}

Ref<ApplePayButtonPart> ApplePayButtonPart::create(ApplePayButtonType buttonType, ApplePayButtonStyle buttonStyle, const String& locale)
{
    return adoptRef(*new ApplePayButtonPart(buttonType, buttonStyle, locale));
}

ApplePayButtonPart::ApplePayButtonPart(ApplePayButtonType buttonType, ApplePayButtonStyle buttonStyle, const String& locale)
    : ControlPart(StyleAppearance::ApplePayButton)
    , m_buttonType(buttonType)
    , m_buttonStyle(buttonStyle)
    , m_locale(locale)
{
}

std::unique_ptr<PlatformControl> ApplePayButtonPart::createPlatformControl()
{
    return controlFactory().createPlatformApplePayButton(*this);
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
