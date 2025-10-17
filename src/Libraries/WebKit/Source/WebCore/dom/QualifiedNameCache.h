/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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

#include "HTMLNames.h"
#include "QualifiedName.h"
#include "SVGNames.h"
#include "XLinkNames.h"
#include "XMLNSNames.h"
#include "XMLNames.h"
#include <wtf/HashSet.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>

#if ENABLE(MATHML)
#include "MathMLNames.h"
#endif

enum class Namespace : uint8_t;
enum class NodeName : uint16_t;

namespace WebCore {

class QualifiedNameCache {
    WTF_MAKE_TZONE_ALLOCATED(QualifiedNameCache);
public:
    QualifiedNameCache() = default;

    Ref<QualifiedName::QualifiedNameImpl> getOrCreate(const QualifiedNameComponents&);
    Ref<QualifiedName::QualifiedNameImpl> getOrCreate(const QualifiedNameComponents&, Namespace, NodeName);
    void remove(QualifiedName::QualifiedNameImpl&);

private:
    static const int staticQualifiedNamesCount = HTMLNames::HTMLTagsCount + HTMLNames::HTMLAttrsCount
#if ENABLE(MATHML)
        + MathMLNames::MathMLTagsCount + MathMLNames::MathMLAttrsCount
#endif
        + SVGNames::SVGTagsCount + SVGNames::SVGAttrsCount
        + XLinkNames::XLinkAttrsCount
        + XMLNSNames::XMLNSAttrsCount
        + XMLNames::XMLAttrsCount;

    struct QualifiedNameHashTraits : public HashTraits<QualifiedName::QualifiedNameImpl*> {
        static const int minimumTableSize = WTF::HashTableCapacityForSize<staticQualifiedNamesCount>::value;
    };

    using QNameSet = UncheckedKeyHashSet<QualifiedName::QualifiedNameImpl*, QualifiedNameHash, QualifiedNameHashTraits>;
    QNameSet m_cache;
};

}
