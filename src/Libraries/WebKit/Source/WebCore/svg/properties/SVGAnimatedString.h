/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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

#include "SVGAnimatedPrimitiveProperty.h"

namespace WebCore {

class TrustedScriptURL;

enum class IsHrefProperty : bool {
    No,
    Yes,
};

using StringOrTrustedScriptURL = std::variant<String, RefPtr<TrustedScriptURL>>;

class SVGAnimatedString : public SVGAnimatedPrimitiveProperty<String> {
public:
    static Ref<SVGAnimatedString> create(SVGElement* contextElement, const IsHrefProperty& isHrefProperty = IsHrefProperty::No)
    {
        return adoptRef(*new SVGAnimatedString(contextElement, isHrefProperty));
    }

    virtual ExceptionOr<void> setBaseVal(const StringOrTrustedScriptURL&);

protected:
    SVGAnimatedString(SVGElement* contextElement, const IsHrefProperty& isHrefProperty = IsHrefProperty::No)
        : SVGAnimatedPrimitiveProperty<String>(contextElement)
        , m_isHrefProperty(isHrefProperty)
    {

    }

private:
    IsHrefProperty m_isHrefProperty;
};

}
