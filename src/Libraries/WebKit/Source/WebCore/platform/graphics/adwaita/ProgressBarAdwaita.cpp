/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#include "ProgressBarAdwaita.h"

#include "GraphicsContextStateSaver.h"
#include "ProgressBarPart.h"

#if USE(THEME_ADWAITA)

namespace WebCore {
using namespace WebCore::Adwaita;

ProgressBarAdwaita::ProgressBarAdwaita(ControlPart& part, ControlFactoryAdwaita& controlFactory)
    : ControlAdwaita(part, controlFactory)
{
}

static double currentAnimationProgress(Seconds animationStartTime)
{
    auto duration = progressAnimationDuration;
    return fmod((MonotonicTime::now().secondsSinceEpoch() - animationStartTime).seconds(), duration.seconds()) / duration.seconds();
}

void ProgressBarAdwaita::draw(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float /*deviceScaleFactor*/, const ControlStyle& style)
{
    GraphicsContextStateSaver stateSaver(graphicsContext);

    SRGBA<uint8_t> progressBarBackgroundColor;

    if (style.states.contains(ControlStyle::State::DarkAppearance))
        progressBarBackgroundColor = progressBarBackgroundColorDark;
    else
        progressBarBackgroundColor = progressBarBackgroundColorLight;

    FloatRect fieldRect = borderRect.rect();
    FloatSize corner(3, 3);
    Path path;

    path.addRoundedRect(fieldRect, corner);
    graphicsContext.setFillRule(WindRule::NonZero);
    graphicsContext.setFillColor(progressBarBackgroundColor);
    graphicsContext.fillPath(path);
    path.clear();

    auto& progressBarPart = owningProgressBarPart();
    bool isDeterminate = progressBarPart.position() >= 0;
    if (isDeterminate) {
        auto progressWidth = fieldRect.width() * progressBarPart.position();
        if (style.states.contains(ControlStyle::State::InlineFlippedWritingMode))
            fieldRect.move(fieldRect.width() - progressWidth, 0);
        fieldRect.setWidth(progressWidth);
    } else {
        double animationProgress = currentAnimationProgress(progressBarPart.animationStartTime());

        // Never let the progress rect shrink smaller than 2 pixels.
        fieldRect.setWidth(std::max<float>(2, fieldRect.width() / progressActivityBlocks));
        auto movableWidth = borderRect.rect().width() - fieldRect.width();

        // We want the first 0.5 units of the animation progress to represent the
        // forward motion and the second 0.5 units to represent the backward motion,
        // thus we multiply by two here to get the full sweep of the progress bar with
        // each direction.
        if (animationProgress < 0.5)
            fieldRect.move(animationProgress * 2 * movableWidth, 0);
        else
            fieldRect.move((1.0 - animationProgress) * 2 * movableWidth, 0);
    }

    path.addRoundedRect(fieldRect, corner);
    graphicsContext.setFillRule(WindRule::NonZero);

    graphicsContext.setFillColor(accentColor(style));
    graphicsContext.fillPath(path);
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
