/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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

#if ENABLE(APPLE_PAY)

#include "ControlPart.h"

namespace WebCore {

enum class ApplePayButtonType : uint8_t {
    Plain,
    Buy,
    SetUp,
    Donate,
    CheckOut,
    Book,
    Subscribe,
#if ENABLE(APPLE_PAY_NEW_BUTTON_TYPES)
    Reload,
    AddMoney,
    TopUp,
    Order,
    Rent,
    Support,
    Contribute,
    Tip,
#endif // ENABLE(APPLE_PAY_NEW_BUTTON_TYPES)
};

enum class ApplePayButtonStyle : uint8_t {
    White,
    WhiteOutline,
    Black,
};

class ApplePayButtonPart : public ControlPart {
public:
    static Ref<ApplePayButtonPart> create();
    WEBCORE_EXPORT static Ref<ApplePayButtonPart> create(ApplePayButtonType, ApplePayButtonStyle, const String& locale);

    ApplePayButtonType buttonType() const { return m_buttonType; }
    void setButtonType(ApplePayButtonType buttonType) { m_buttonType = buttonType; }

    ApplePayButtonStyle buttonStyle() const { return m_buttonStyle; }
    void setButtonStyle(ApplePayButtonStyle buttonStyle) { m_buttonStyle = buttonStyle; }

    String locale() const { return m_locale; }
    void setLocale(String locale) { m_locale = locale; }

private:
    ApplePayButtonPart(ApplePayButtonType, ApplePayButtonStyle, const String& locale);

    std::unique_ptr<PlatformControl> createPlatformControl() override;

    ApplePayButtonType m_buttonType;
    ApplePayButtonStyle m_buttonStyle;
    String m_locale;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CONTROL_PART(ApplePayButton)

#endif // ENABLE(APPLE_PAY)
