/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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

#include "FilterOperations.h"
#include "GraphicsTypes.h"
#include "PlatformLayer.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS NSValue;

#if PLATFORM(MAC)
OBJC_CLASS CAPresentationModifier;
OBJC_CLASS CAPresentationModifierGroup;
#endif

namespace WebCore {

class PlatformCALayer;

#if PLATFORM(MAC)
using TypedFilterPresentationModifier = std::pair<FilterOperation::Type, RetainPtr<CAPresentationModifier>>;
#endif

class PlatformCAFilters {
public:
    WEBCORE_EXPORT static void setFiltersOnLayer(PlatformLayer*, const FilterOperations&, bool backdropIsOpaque);
    WEBCORE_EXPORT static void setBlendingFiltersOnLayer(PlatformLayer*, const BlendMode);
    static bool isAnimatedFilterProperty(FilterOperation::Type);
    static String animatedFilterPropertyName(FilterOperation::Type);
    static bool isValidAnimatedFilterPropertyName(const String&);

    WEBCORE_EXPORT static RetainPtr<NSValue> filterValueForOperation(const FilterOperation&);

    // A null operation indicates that we should make a "no-op" filter of the given type.
    static RetainPtr<NSValue> colorMatrixValueForFilter(FilterOperation::Type, const FilterOperation*);

#if PLATFORM(MAC)
    WEBCORE_EXPORT static void presentationModifiers(const FilterOperations& initialFilters, const FilterOperations* canonicalFilters, Vector<TypedFilterPresentationModifier>& presentationModifiers, RetainPtr<CAPresentationModifierGroup>&);
    WEBCORE_EXPORT static void updatePresentationModifiers(const FilterOperations& filters, const Vector<TypedFilterPresentationModifier>& presentationModifiers);
    WEBCORE_EXPORT static size_t presentationModifierCount(const FilterOperations&);
#endif
};

}
