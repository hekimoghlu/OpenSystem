/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#include "AppleVisualEffect.h"

#if HAVE(CORE_MATERIAL)

#include <wtf/text/TextStream.h>

namespace WebCore {

bool appleVisualEffectNeedsBackdrop(AppleVisualEffect effect)
{
    switch (effect) {
    case AppleVisualEffect::BlurUltraThinMaterial:
    case AppleVisualEffect::BlurThinMaterial:
    case AppleVisualEffect::BlurMaterial:
    case AppleVisualEffect::BlurThickMaterial:
    case AppleVisualEffect::BlurChromeMaterial:
        return true;
    case AppleVisualEffect::None:
    case AppleVisualEffect::VibrancyLabel:
    case AppleVisualEffect::VibrancySecondaryLabel:
    case AppleVisualEffect::VibrancyTertiaryLabel:
    case AppleVisualEffect::VibrancyQuaternaryLabel:
    case AppleVisualEffect::VibrancyFill:
    case AppleVisualEffect::VibrancySecondaryFill:
    case AppleVisualEffect::VibrancyTertiaryFill:
    case AppleVisualEffect::VibrancySeparator:
        return false;
    }

    ASSERT_NOT_REACHED();
    return false;
}

TextStream& operator<<(TextStream& ts, AppleVisualEffect effect)
{
    switch (effect) {
    case AppleVisualEffect::None:
        ts << "none";
        break;
    case AppleVisualEffect::BlurUltraThinMaterial:
        ts << "blur-material-ultra-thin";
        break;
    case AppleVisualEffect::BlurThinMaterial:
        ts << "blur-material-thin";
        break;
    case AppleVisualEffect::BlurMaterial:
        ts << "blur-material";
        break;
    case AppleVisualEffect::BlurThickMaterial:
        ts << "blur-material-thick";
        break;
    case AppleVisualEffect::BlurChromeMaterial:
        ts << "blur-material-chrome";
        break;
    case AppleVisualEffect::VibrancyLabel:
        ts << "vibrancy-label";
        break;
    case AppleVisualEffect::VibrancySecondaryLabel:
        ts << "vibrancy-secondary-label";
        break;
    case AppleVisualEffect::VibrancyTertiaryLabel:
        ts << "vibrancy-tertiary-label";
        break;
    case AppleVisualEffect::VibrancyQuaternaryLabel:
        ts << "vibrancy-quaternary-label";
        break;
    case AppleVisualEffect::VibrancyFill:
        ts << "vibrancy-fill";
        break;
    case AppleVisualEffect::VibrancySecondaryFill:
        ts << "vibrancy-secondary-fill";
        break;
    case AppleVisualEffect::VibrancyTertiaryFill:
        ts << "vibrancy-tertiary-fill";
        break;
    case AppleVisualEffect::VibrancySeparator:
        ts << "vibrancy-separator";
        break;
    }
    return ts;
}

} // namespace WebCore

#endif // HAVE(CORE_MATERIAL)
