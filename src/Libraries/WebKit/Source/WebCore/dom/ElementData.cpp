/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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
#include "ElementData.h"

#include "Attr.h"
#include "Element.h"
#include "HTMLNames.h"
#include "ImmutableStyleProperties.h"
#include "MutableStyleProperties.h"
#include "StyleProperties.h"
#include "StylePropertiesInlines.h"
#include "XMLNames.h"
#include <wtf/ZippedRange.h>

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ElementData);
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ShareableElementData);

void ElementData::destroy()
{
    if (auto* uniqueData = dynamicDowncast<UniqueElementData>(*this))
        delete uniqueData;
    else
        delete uncheckedDowncast<ShareableElementData>(this);
}

ElementData::ElementData()
    : m_arraySizeAndFlags(s_flagIsUnique)
{
}

ElementData::ElementData(unsigned arraySize)
    : m_arraySizeAndFlags(arraySize << s_flagCount)
{
}

struct SameSizeAsElementData : public RefCounted<SameSizeAsElementData> {
    unsigned bitfield;
    void* refPtrs[3];
};

static_assert(sizeof(ElementData) == sizeof(SameSizeAsElementData), "element attribute data should stay small");

static size_t sizeForShareableElementDataWithAttributeCount(unsigned count)
{
    return sizeof(ShareableElementData) + sizeof(Attribute) * count;
}

Ref<ShareableElementData> ShareableElementData::createWithAttributes(std::span<const Attribute> attributes)
{
    void* slot = ShareableElementDataMalloc::malloc(sizeForShareableElementDataWithAttributeCount(attributes.size()));
    return adoptRef(*new (NotNull, slot) ShareableElementData(attributes));
}

Ref<UniqueElementData> UniqueElementData::create()
{
    return adoptRef(*new UniqueElementData);
}

ShareableElementData::ShareableElementData(std::span<const Attribute> attributes)
    : ElementData(attributes.size())
{
    for (auto [sourceAttribute, destinationAttribute] : zippedRange(attributes, this->attributes()))
        new (NotNull, &destinationAttribute) Attribute(sourceAttribute);
}

ShareableElementData::~ShareableElementData()
{
    for (auto& attribute : attributes())
        attribute.~Attribute();
}

ShareableElementData::ShareableElementData(const UniqueElementData& other)
    : ElementData(other, false)
{
    ASSERT(!other.m_presentationalHintStyle);

    if (other.m_inlineStyle) {
        ASSERT(!other.m_inlineStyle->hasCSSOMWrapper());
        m_inlineStyle = other.m_inlineStyle->immutableCopyIfNeeded();
    }

    for (auto [sourceAttribute, destinationAttribute] : zippedRange(other.m_attributeVector.span(), attributes()))
        new (NotNull, &destinationAttribute) Attribute(sourceAttribute);
}

inline uint32_t ElementData::arraySizeAndFlagsFromOther(const ElementData& other, bool isUnique)
{
    if (isUnique) {
        // Set isUnique and ignore arraySize.
        return (other.m_arraySizeAndFlags | s_flagIsUnique) & s_flagsMask;
    }
    // Clear isUnique and set arraySize.
    return (other.m_arraySizeAndFlags & (s_flagsMask & ~s_flagIsUnique)) | other.length() << s_flagCount;
}

ElementData::ElementData(const ElementData& other, bool isUnique)
    : m_arraySizeAndFlags(ElementData::arraySizeAndFlagsFromOther(other, isUnique))
    , m_classNames(other.m_classNames)
    , m_idForStyleResolution(other.m_idForStyleResolution)
{
    // NOTE: The inline style is copied by the subclass copy constructor since we don't know what to do with it here.
}

UniqueElementData::UniqueElementData()
{
}

UniqueElementData::UniqueElementData(const UniqueElementData& other)
    : ElementData(other, true)
    , m_presentationalHintStyle(other.m_presentationalHintStyle)
    , m_attributeVector(other.m_attributeVector)
{
    if (other.m_inlineStyle)
        m_inlineStyle = other.m_inlineStyle->mutableCopy();
}

UniqueElementData::UniqueElementData(const ShareableElementData& other)
    : ElementData(other, true)
    , m_attributeVector(other.attributes())
{
    // An ShareableElementData should never have a mutable inline StyleProperties attached.
    ASSERT(!other.m_inlineStyle || !other.m_inlineStyle->isMutable());
    m_inlineStyle = other.m_inlineStyle;
}

Ref<UniqueElementData> ElementData::makeUniqueCopy() const
{
    if (auto* uniqueData = dynamicDowncast<const UniqueElementData>(*this))
        return adoptRef(*new UniqueElementData(*uniqueData));
    return adoptRef(*new UniqueElementData(downcast<const ShareableElementData>(*this)));
}

Ref<ShareableElementData> UniqueElementData::makeShareableCopy() const
{
    void* slot = ShareableElementDataMalloc::malloc(sizeForShareableElementDataWithAttributeCount(m_attributeVector.size()));
    return adoptRef(*new (NotNull, slot) ShareableElementData(*this));
}

bool ElementData::isEquivalent(const ElementData* other) const
{
    if (!other)
        return isEmpty();

    if (length() != other->length())
        return false;

    for (auto& attribute : attributes()) {
        auto* otherAttr = other->findAttributeByName(attribute.name());
        if (!otherAttr || attribute.value() != otherAttr->value())
            return false;
    }

    return true;
}

Attribute* UniqueElementData::findAttributeByName(const QualifiedName& name)
{
    for (auto& attribute : m_attributeVector) {
        if (attribute.name().matches(name))
            return &attribute;
    }
    return nullptr;
}

}
