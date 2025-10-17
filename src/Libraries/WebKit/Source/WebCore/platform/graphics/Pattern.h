/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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

#include "AffineTransform.h"
#include "SourceImage.h"

#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

#if USE(CG)
typedef struct CGPattern* CGPatternRef;
typedef RetainPtr<CGPatternRef> PlatformPatternPtr;
#elif USE(CAIRO)
typedef struct _cairo_pattern cairo_pattern_t;
typedef cairo_pattern_t* PlatformPatternPtr;
#elif USE(SKIA)
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkShader.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
typedef sk_sp<SkShader> PlatformPatternPtr;
#endif

namespace WebCore {

class GraphicsContext;

class Pattern final : public ThreadSafeRefCounted<Pattern> {
public:
    struct Parameters {
        Parameters(bool repeatX = true, bool repeatY = true, AffineTransform patternSpaceTransform = { })
            : repeatX(repeatX)
            , repeatY(repeatY)
            , patternSpaceTransform(patternSpaceTransform)
        {
        }
        bool repeatX;
        bool repeatY;
        AffineTransform patternSpaceTransform;
    };

    WEBCORE_EXPORT static Ref<Pattern> create(SourceImage&& tileImage, const Parameters& = { });
    WEBCORE_EXPORT ~Pattern();

    WEBCORE_EXPORT const SourceImage& tileImage() const;
    WEBCORE_EXPORT void setTileImage(SourceImage&&);

    RefPtr<NativeImage> tileNativeImage() const;
    RefPtr<ImageBuffer> tileImageBuffer() const;

    const Parameters& parameters() const { return m_parameters; }

    // Pattern space is an abstract space that maps to the default user space by the transformation 'userSpaceTransform'
#if USE(SKIA)
    PlatformPatternPtr createPlatformPattern(const AffineTransform& userSpaceTransform, const SkSamplingOptions&) const;
#else
    PlatformPatternPtr createPlatformPattern(const AffineTransform& userSpaceTransform) const;
#endif

    void setPatternSpaceTransform(const AffineTransform&);

    const AffineTransform& patternSpaceTransform() const { return m_parameters.patternSpaceTransform; };
    bool repeatX() const { return m_parameters.repeatX; }
    bool repeatY() const { return m_parameters.repeatY; }

private:
    Pattern(SourceImage&&, const Parameters&);

    SourceImage m_tileImage;
    Parameters m_parameters;
};

} //namespace
