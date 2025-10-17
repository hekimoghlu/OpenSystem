/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
#include "ISOSchemeTypeBox.h"

#include <JavaScriptCore/DataView.h>

using JSC::DataView;

namespace WebCore {

ISOSchemeTypeBox::ISOSchemeTypeBox() = default;
ISOSchemeTypeBox::~ISOSchemeTypeBox() = default;

bool ISOSchemeTypeBox::parse(DataView& view, unsigned& offset)
{
    if (!ISOFullBox::parse(view, offset))
        return false;

    if (!checkedRead<uint32_t>(m_schemeType, view, offset, BigEndian))
        return false;

    if (!checkedRead<uint32_t>(m_schemeVersion, view, offset, BigEndian))
        return false;

    return true;
}

}
