/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

#include "ISOBox.h"

namespace WebCore {

class WEBCORE_EXPORT ISOSchemeTypeBox final : public ISOFullBox {
public:
    ISOSchemeTypeBox();
    ~ISOSchemeTypeBox();

    static FourCC boxTypeName() { return std::span { "schm" }; }

    FourCC schemeType() const { return m_schemeType; }
    uint32_t schemeVersion() const { return m_schemeVersion; }

    bool parse(JSC::DataView&, unsigned& offset) override;

private:
    FourCC m_schemeType;
    uint32_t m_schemeVersion { 0 };
};

}

SPECIALIZE_TYPE_TRAITS_ISOBOX(ISOSchemeTypeBox)
