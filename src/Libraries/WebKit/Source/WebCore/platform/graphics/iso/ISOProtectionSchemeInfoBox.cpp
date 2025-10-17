/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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
#include "ISOProtectionSchemeInfoBox.h"

#include "ISOSchemeInformationBox.h"
#include "ISOSchemeTypeBox.h"
#include <JavaScriptCore/DataView.h>

using JSC::DataView;

namespace WebCore {

ISOProtectionSchemeInfoBox::ISOProtectionSchemeInfoBox() = default;
ISOProtectionSchemeInfoBox::~ISOProtectionSchemeInfoBox() = default;

bool ISOProtectionSchemeInfoBox::parse(DataView& view, unsigned& offset)
{
    unsigned localOffset = offset;
    if (!ISOBox::parse(view, localOffset))
        return false;

    if (!m_originalFormatBox.read(view, localOffset))
        return false;

    if (localOffset - offset == m_size) {
        offset = localOffset;
        return true;
    }

    auto optionalBoxType = ISOBox::peekBox(view, localOffset);
    if (!optionalBoxType)
        return false;

    if (optionalBoxType.value().first == ISOSchemeTypeBox::boxTypeName()) {
        m_schemeTypeBox = makeUnique<ISOSchemeTypeBox>();
        if (!m_schemeTypeBox->read(view, localOffset))
            return false;

        if (localOffset - offset == m_size) {
            offset = localOffset;
            return true;
        }

        optionalBoxType = ISOBox::peekBox(view, localOffset);
        if (!optionalBoxType)
            return false;
    }

    if (optionalBoxType.value().first == ISOSchemeInformationBox::boxTypeName()) {
        m_schemeInformationBox = makeUnique<ISOSchemeInformationBox>();
        if (!m_schemeInformationBox->read(view, localOffset))
            return false;

        if (localOffset - offset != m_size)
            return false;
    }

    offset = localOffset;
    return true;
}

}
