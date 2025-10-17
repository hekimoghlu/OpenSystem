/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#include <wtf/Forward.h>
#include <wtf/OptionSet.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class CSSValue;
class Element;

namespace CQ {

namespace FeatureSchemas {
const MQ::FeatureSchema& width();
const MQ::FeatureSchema& height();
const MQ::FeatureSchema& inlineSize();
const MQ::FeatureSchema& blockSize();
const MQ::FeatureSchema& aspectRatio();
const MQ::FeatureSchema& orientation();
};

enum class Axis : uint8_t {
    Block   = 1 << 0,
    Inline  = 1 << 1,
    Width   = 1 << 2,
    Height  = 1 << 3,
};
OptionSet<Axis> requiredAxesForFeature(const MQ::Feature&);

enum class ContainsUnknownFeature : bool { No, Yes };

struct ContainerQuery {
    AtomString name;
    MQ::Condition condition;
    OptionSet<CQ::Axis> requiredAxes;
    ContainsUnknownFeature containsUnknownFeature;
};

void serialize(StringBuilder&, const ContainerQuery&);

}

}
