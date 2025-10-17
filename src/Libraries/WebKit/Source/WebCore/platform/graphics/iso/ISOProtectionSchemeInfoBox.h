/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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

#include "ISOOriginalFormatBox.h"

namespace WebCore {

class ISOSchemeTypeBox;
class ISOSchemeInformationBox;

class WEBCORE_EXPORT ISOProtectionSchemeInfoBox final : public ISOBox {
public:
    ISOProtectionSchemeInfoBox();
    ~ISOProtectionSchemeInfoBox();

    static FourCC boxTypeName() { return std::span { "sinf" }; }

    const ISOOriginalFormatBox& originalFormatBox() const { return m_originalFormatBox; }
    const ISOSchemeTypeBox* schemeTypeBox() const { return m_schemeTypeBox.get(); }
    const ISOSchemeInformationBox* schemeInformationBox() const { return m_schemeInformationBox.get(); }

    bool parse(JSC::DataView&, unsigned& offset) override;

private:
    ISOOriginalFormatBox m_originalFormatBox;
    std::unique_ptr<ISOSchemeTypeBox> m_schemeTypeBox;
    std::unique_ptr<ISOSchemeInformationBox> m_schemeInformationBox;
};

}

SPECIALIZE_TYPE_TRAITS_ISOBOX(ISOProtectionSchemeInfoBox)
