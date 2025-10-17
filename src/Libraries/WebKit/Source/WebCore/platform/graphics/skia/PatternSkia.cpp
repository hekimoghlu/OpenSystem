/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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
#include "Pattern.h"

#if USE(SKIA)
#include "AffineTransform.h"
#include "ImageBuffer.h"
#include "NativeImage.h"
#include <skia/core/SkImage.h>
#include <skia/core/SkSamplingOptions.h>
#include <skia/core/SkTileMode.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkMatrix.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

sk_sp<SkShader> Pattern::createPlatformPattern(const AffineTransform&, const SkSamplingOptions& samplingOptions) const
{
    auto nativeImage = tileNativeImage();
    if (!nativeImage)
        return nullptr;

    auto platformImage = nativeImage->platformImage();
    if (!platformImage)
        return nullptr;

    return platformImage->makeShader(repeatX() ? SkTileMode::kRepeat : SkTileMode::kDecal, repeatY() ? SkTileMode::kRepeat : SkTileMode::kDecal, samplingOptions, patternSpaceTransform());
}

} // namespace WebCore

#endif // USE(SKIA)
