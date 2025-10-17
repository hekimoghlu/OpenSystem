/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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

#include "MediaQuery.h"

namespace WebCore {

namespace Calculation {
enum class Category : uint8_t;
}

struct ComputedStyleDependencies;

namespace MQ {

// Interface exposed by schemas that can provide a value for the media-progress() function.
struct MediaProgressProviding {
    virtual ~MediaProgressProviding();

    virtual AtomString name() const = 0;
    virtual Calculation::Category category() const = 0;
    virtual void collectComputedStyleDependencies(ComputedStyleDependencies&) const = 0;

    virtual double valueInCanonicalUnits(const FeatureEvaluationContext&) const = 0;
};

namespace Features {

const FeatureSchema& animation();
const FeatureSchema& anyHover();
const FeatureSchema& anyPointer();
const FeatureSchema& aspectRatio();
const FeatureSchema& color();
const FeatureSchema& colorGamut();
const FeatureSchema& colorIndex();
const FeatureSchema& deviceAspectRatio();
const FeatureSchema& deviceHeight();
const FeatureSchema& devicePixelRatio();
const FeatureSchema& deviceWidth();
const FeatureSchema& dynamicRange();
const FeatureSchema& forcedColors();
const FeatureSchema& grid();
const FeatureSchema& height();
const FeatureSchema& hover();
const FeatureSchema& invertedColors();
const FeatureSchema& monochrome();
const FeatureSchema& orientation();
const FeatureSchema& overflowBlock();
const FeatureSchema& overflowInline();
const FeatureSchema& pointer();
const FeatureSchema& prefersContrast();
const FeatureSchema& prefersDarkInterface();
const FeatureSchema& prefersReducedMotion();
const FeatureSchema& resolution();
const FeatureSchema& scan();
const FeatureSchema& scripting();
const FeatureSchema& transform2d();
const FeatureSchema& transform3d();
const FeatureSchema& transition();
const FeatureSchema& update();
const FeatureSchema& videoPlayableInline();
const FeatureSchema& width();
#if ENABLE(APPLICATION_MANIFEST)
const FeatureSchema& displayMode();
#endif
#if ENABLE(DARK_MODE_CSS)
const FeatureSchema& prefersColorScheme();
#endif

Vector<const FeatureSchema*> allSchemas();
Vector<const MediaProgressProviding*> allMediaProgressProvidingSchemas();

} // namespace Features
} // namespace MQ
} // namespace WebCore
