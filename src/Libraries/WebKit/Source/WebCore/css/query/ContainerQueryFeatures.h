/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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

#include "GenericMediaQueryTypes.h"

namespace WebCore {

class CSSToLengthConversionData;
class RenderBox;
class RenderStyle;
class RenderView;

namespace Calculation {
enum class Category : uint8_t;
}

namespace CQ {

// Interface exposed by schemas that can provide a value for the container-progress() function.
struct ContainerProgressProviding {
    virtual ~ContainerProgressProviding();

    virtual AtomString name() const = 0;
    virtual WebCore::Calculation::Category category() const = 0;
    virtual void collectComputedStyleDependencies(ComputedStyleDependencies&) const = 0;

    virtual double valueInCanonicalUnits(const RenderBox&) const = 0;
    virtual double valueInCanonicalUnits(const RenderView&, const RenderStyle&) const = 0;
};

namespace Features {

const MQ::FeatureSchema& width();
const MQ::FeatureSchema& height();
const MQ::FeatureSchema& inlineSize();
const MQ::FeatureSchema& blockSize();
const MQ::FeatureSchema& aspectRatio();
const MQ::FeatureSchema& orientation();
const MQ::FeatureSchema& style();

Vector<const MQ::FeatureSchema*> allSchemas();
Vector<const ContainerProgressProviding*> allContainerProgressProvidingSchemas();

} // namespace Features
} // namespace CQ
} // namespace WebCore
