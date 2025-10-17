/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

#include "SVGPoint.h"
#include "SVGValuePropertyList.h"

namespace WebCore {

class SVGPointList final : public SVGValuePropertyList<SVGPoint> {
    using Base = SVGValuePropertyList<SVGPoint>;
    using Base::Base;

public:
    static Ref<SVGPointList> create()
    {
        return adoptRef(*new SVGPointList());
    }

    static Ref<SVGPointList> create(SVGPropertyOwner* owner, SVGPropertyAccess access)
    {
        return adoptRef(*new SVGPointList(owner, access));
    }

    static Ref<SVGPointList> create(const SVGPointList& other, SVGPropertyAccess access)
    {
        return adoptRef(*new SVGPointList(other, access));
    }

    bool parse(StringView);
    String valueAsString() const override;
};

}
