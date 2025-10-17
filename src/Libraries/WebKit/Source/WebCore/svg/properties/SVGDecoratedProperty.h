/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

#include <wtf/RefCounted.h>

namespace WebCore {

template<typename DecorationType>
class SVGDecoratedProperty : public RefCounted<SVGDecoratedProperty<DecorationType>> {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(SVGDecoratedProperty);
public:
    SVGDecoratedProperty() = default;
    virtual ~SVGDecoratedProperty() = default;

    virtual void setValueInternal(const DecorationType&) = 0;
    virtual bool setValue(const DecorationType& value)
    {
        setValueInternal(value);
        return true;
    }

    // Used internally. It doesn't check for highestExposedEnumValue for example.
    virtual DecorationType valueInternal() const = 0;

    // Used by the DOM APIs.
    virtual DecorationType value() const { return valueInternal(); }

    virtual String valueAsString() const = 0;
    virtual Ref<SVGDecoratedProperty<DecorationType>> clone() = 0;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename DecorationType>, SVGDecoratedProperty<DecorationType>);

} // namespace WebCore
