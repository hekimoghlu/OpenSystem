/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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
#include "DocumentSharedObjectPool.h"

#include "Element.h"
#include "ElementData.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DocumentSharedObjectPool);

struct DocumentSharedObjectPool::ShareableElementDataHash {
    static unsigned hash(const Ref<ShareableElementData>& data)
    {
        return computeHash(data->attributes());
    }
    static bool equal(const Ref<ShareableElementData>& a, const Ref<ShareableElementData>& b)
    {
        // We need to disable type checking because std::has_unique_object_representations_v<Attribute>
        // return false. Attribute contains pointers but memcmp() is safe because those pointers were
        // atomized.
        return equalSpans<WTF::IgnoreTypeChecks::Yes>(a->attributes(), b->attributes());
    }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

struct AttributeSpanTranslator {
    static unsigned hash(std::span<const Attribute> attributes)
    {
        return computeHash(attributes);
    }

    static bool equal(const Ref<ShareableElementData>& a, std::span<const Attribute> b)
    {
        // We need to disable type checking because std::has_unique_object_representations_v<Attribute>
        // return false. Attribute contains pointers but memcmp() is safe because those pointers were
        // atomized.
        return equalSpans<WTF::IgnoreTypeChecks::Yes>(a->attributes(), b);
    }

    static void translate(Ref<ShareableElementData>& location, std::span<const Attribute> attributes, unsigned /*hash*/)
    {
        location = ShareableElementData::createWithAttributes(attributes);
    }
};

Ref<ShareableElementData> DocumentSharedObjectPool::cachedShareableElementDataWithAttributes(std::span<const Attribute> attributes)
{
    ASSERT(!attributes.empty());

    return m_shareableElementDataCache.add<AttributeSpanTranslator>(attributes).iterator->get();
}

} // namespace WebCore
