/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
#import "ApplePayButtonCocoa.h"

#if ENABLE(APPLE_PAY)

#import "ApplePayButtonPart.h"
#import "FloatRoundedRect.h"
#import "GraphicsContextCG.h"
#import <wtf/TZoneMallocInlines.h>
#import <pal/cocoa/PassKitSoftLink.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ApplePayButtonCocoa);

ApplePayButtonCocoa::ApplePayButtonCocoa(ApplePayButtonPart& owningPart)
    : PlatformControl(owningPart)
{
}

static PKPaymentButtonType toPKPaymentButtonType(ApplePayButtonType type)
{
    switch (type) {
    case ApplePayButtonType::Plain:
        return PKPaymentButtonTypePlain;
    case ApplePayButtonType::Buy:
        return PKPaymentButtonTypeBuy;
    case ApplePayButtonType::SetUp:
        return PKPaymentButtonTypeSetUp;
    case ApplePayButtonType::Donate:
        return PKPaymentButtonTypeDonate;
    case ApplePayButtonType::CheckOut:
        return PKPaymentButtonTypeCheckout;
    case ApplePayButtonType::Book:
        return PKPaymentButtonTypeBook;
    case ApplePayButtonType::Subscribe:
        return PKPaymentButtonTypeSubscribe;
#if HAVE(PASSKIT_NEW_BUTTON_TYPES)
    case ApplePayButtonType::Reload:
        return PKPaymentButtonTypeReload;
    case ApplePayButtonType::AddMoney:
        return PKPaymentButtonTypeAddMoney;
    case ApplePayButtonType::TopUp:
        return PKPaymentButtonTypeTopUp;
    case ApplePayButtonType::Order:
        return PKPaymentButtonTypeOrder;
    case ApplePayButtonType::Rent:
        return PKPaymentButtonTypeRent;
    case ApplePayButtonType::Support:
        return PKPaymentButtonTypeSupport;
    case ApplePayButtonType::Contribute:
        return PKPaymentButtonTypeContribute;
    case ApplePayButtonType::Tip:
        return PKPaymentButtonTypeTip;
#endif // HAVE(PASSKIT_NEW_BUTTON_TYPES)
    }
}

static PKPaymentButtonStyle toPKPaymentButtonStyle(ApplePayButtonStyle style)
{
    switch (style) {
    case ApplePayButtonStyle::White:
        return PKPaymentButtonStyleWhite;
    case ApplePayButtonStyle::WhiteOutline:
        return PKPaymentButtonStyleWhiteOutline;
    case ApplePayButtonStyle::Black:
        return PKPaymentButtonStyleBlack;
    }
}

void ApplePayButtonCocoa::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float, const ControlStyle&)
{
    auto largestCornerRadius = std::max({
        borderRect.radii().topLeft().maxDimension(),
        borderRect.radii().topRight().maxDimension(),
        borderRect.radii().bottomLeft().maxDimension(),
        borderRect.radii().bottomRight().maxDimension()
    });

    GraphicsContextStateSaver stateSaver(context);

    context.setShouldSmoothFonts(true);
    context.scale(FloatSize(1, -1));

    auto logicalRect = borderRect.rect();
    const auto& applePayButtonPart = owningApplePayButtonPart();
    
    PKDrawApplePayButtonWithCornerRadius(
        context.platformContext(),
        CGRectMake(logicalRect.x(), -logicalRect.maxY(), logicalRect.width(), logicalRect.height()),
        1.0,
        largestCornerRadius,
        toPKPaymentButtonType(applePayButtonPart.buttonType()),
        toPKPaymentButtonStyle(applePayButtonPart.buttonStyle()),
        applePayButtonPart.locale()
    );
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
