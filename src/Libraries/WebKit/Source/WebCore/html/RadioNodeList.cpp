/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#include "RadioNodeList.h"

#include "HTMLFormElement.h"
#include "HTMLInputElement.h"
#include "HTMLObjectElement.h"
#include "LiveNodeListInlines.h"
#include "NodeRareData.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RadioNodeList);

RadioNodeList::RadioNodeList(ContainerNode& rootNode, const AtomString& name)
    : CachedLiveNodeList(rootNode, NodeListInvalidationType::InvalidateForFormControls)
    , m_name(name)
    , m_isRootedAtTreeScope(is<HTMLFormElement>(rootNode))
{
}

Ref<RadioNodeList> RadioNodeList::create(ContainerNode& rootNode, const AtomString& name)
{
    return adoptRef(*new RadioNodeList(rootNode, name));
}

RadioNodeList::~RadioNodeList()
{
    ownerNode().nodeLists()->removeCacheWithAtomName(*this, m_name);
}

static RefPtr<HTMLInputElement> nonEmptyRadioButton(Node& node)
{
    auto* inputElement = dynamicDowncast<HTMLInputElement>(node);
    if (!inputElement)
        return nullptr;

    if (!inputElement->isRadioButton() || inputElement->value().isEmpty())
        return nullptr;
    return inputElement;
}

String RadioNodeList::value() const
{
    auto length = this->length();
    for (unsigned i = 0; i < length; ++i) {
        if (auto button = nonEmptyRadioButton(*item(i))) {
            if (button->checked())
                return button->value();
        }
    }
    return String();
}

void RadioNodeList::setValue(const String& value)
{
    auto length = this->length();
    for (unsigned i = 0; i < length; ++i) {
        if (auto button = nonEmptyRadioButton(*item(i))) {
            if (button->value() == value) {
                button->setChecked(true);
                return;
            }
        }
    }
}

bool RadioNodeList::elementMatches(Element& element) const
{
    if (!element.isFormListedElement())
        return false;

    if (auto* input = dynamicDowncast<HTMLInputElement>(element); input && input->isImageButton())
        return false;

    if (is<HTMLFormElement>(ownerNode())) {
        RefPtr form = element.asFormListedElement()->form();
        if (form != &ownerNode())
            return false;
    }

    return element.getIdAttribute() == m_name || element.getNameAttribute() == m_name;
}

} // namespace WebCore
