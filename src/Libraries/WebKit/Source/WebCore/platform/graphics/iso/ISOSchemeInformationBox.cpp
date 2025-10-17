/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#include "ISOSchemeInformationBox.h"

#include "ISOTrackEncryptionBox.h"
#include <JavaScriptCore/DataView.h>

using JSC::DataView;

namespace WebCore {

ISOSchemeInformationBox::ISOSchemeInformationBox() = default;
ISOSchemeInformationBox::~ISOSchemeInformationBox() = default;

bool ISOSchemeInformationBox::parse(DataView& view, unsigned& offset)
{
    unsigned localOffset = offset;
    if (!ISOBox::parse(view, localOffset))
        return false;

    auto schemeSpecificBoxType = ISOBox::peekBox(view, localOffset);
    if (!schemeSpecificBoxType)
        return false;

    if (schemeSpecificBoxType.value().first == ISOTrackEncryptionBox::boxTypeName()) {
        if (localOffset + schemeSpecificBoxType.value().second > offset + m_size)
            return false;

        m_schemeSpecificData = makeUnique<ISOTrackEncryptionBox>();
        if (!m_schemeSpecificData->read(view, localOffset))
            return false;
    }

    return true;
}

}
