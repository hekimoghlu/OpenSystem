/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#import "ImageControlsButtonMac.h"

#if PLATFORM(MAC) && ENABLE(SERVICE_CONTROLS)

#import "ControlFactoryMac.h"
#import "FloatRoundedRect.h"
#import "GraphicsContext.h"
#import "ImageControlsButtonPart.h"
#import "LocalDefaultSystemAppearance.h"
#import <pal/spi/mac/NSServicesRolloverButtonCellSPI.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageControlsButtonMac);

ImageControlsButtonMac::ImageControlsButtonMac(ImageControlsButtonPart& owningPart, ControlFactoryMac& controlFactory, NSServicesRolloverButtonCell *servicesRolloverButtonCell)
    : ControlMac(owningPart, controlFactory)
    , m_servicesRolloverButtonCell(servicesRolloverButtonCell)
{
}

IntSize ImageControlsButtonMac::servicesRolloverButtonCellSize()
{
    auto& controlFactory = ControlFactoryMac::shared();
    if (auto* servicesRolloverButtonCell = controlFactory.servicesRolloverButtonCell())
        return IntSize { [servicesRolloverButtonCell cellSize] };
    return { };
}

void ImageControlsButtonMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    drawCell(context, borderRect.rect(), deviceScaleFactor, style, m_servicesRolloverButtonCell.get(), true);
}

} // namespace WebCore

#endif // PLATFORM(MAC) && ENABLE(SERVICE_CONTROLS)
