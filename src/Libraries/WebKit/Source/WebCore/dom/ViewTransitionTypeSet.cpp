/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
#include "ViewTransitionTypeSet.h"

#include "Document.h"
#include "PseudoClassChangeInvalidation.h"
#include <wtf/IsoMallocInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ViewTransitionTypeSet);

ViewTransitionTypeSet::ViewTransitionTypeSet(Document& document, Vector<AtomString>&& initialActiveTypes)
    : m_typeSet()
    , m_document(document)
{
    for (auto initialActiveType : initialActiveTypes)
        m_typeSet.add(initialActiveType);
}

void ViewTransitionTypeSet::initializeSetLike(DOMSetAdapter& setAdapter) const
{
    for (auto activeType : m_typeSet)
        setAdapter.add<IDLDOMString>(activeType);
}

void ViewTransitionTypeSet::clearFromSetLike()
{
    std::optional<Style::PseudoClassChangeInvalidation> styleInvalidation;
    if (!m_document)
        return;
    if (m_document->documentElement()) {
        styleInvalidation.emplace(
            *m_document->documentElement(),
            CSSSelector::PseudoClass::ActiveViewTransitionType,
            Style::PseudoClassChangeInvalidation::AnyValue
        );
    }

    m_typeSet.clear();
}

void ViewTransitionTypeSet::addToSetLike(const AtomString& type)
{
    std::optional<Style::PseudoClassChangeInvalidation> styleInvalidation;
    if (!m_document)
        return;
    if (m_document->documentElement()) {
        styleInvalidation.emplace(
            *m_document->documentElement(),
            CSSSelector::PseudoClass::ActiveViewTransitionType,
            Style::PseudoClassChangeInvalidation::AnyValue
        );
    }

    m_typeSet.add(type);
}

bool ViewTransitionTypeSet::removeFromSetLike(const AtomString& type)
{
    std::optional<Style::PseudoClassChangeInvalidation> styleInvalidation;
    if (!m_document)
        return false;
    if (m_document->documentElement()) {
        styleInvalidation.emplace(
            *m_document->documentElement(),
            CSSSelector::PseudoClass::ActiveViewTransitionType,
            Style::PseudoClassChangeInvalidation::AnyValue
        );
    }

    return m_typeSet.remove(type);
}

bool ViewTransitionTypeSet::hasType(const AtomString& type) const
{
    return m_typeSet.contains(type);
}

}
