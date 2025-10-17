/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
#include "SVGLength.h"

#include "SVGElement.h"

namespace WebCore {

ExceptionOr<float> SVGLength::valueForBindings()
{
    return m_value.valueForBindings(SVGLengthContext { RefPtr { contextElement() }.get() });
}

ExceptionOr<void> SVGLength::setValueForBindings(float value)
{
    if (isReadOnly())
        return Exception { ExceptionCode::NoModificationAllowedError };

    auto result = m_value.setValue(SVGLengthContext { RefPtr { contextElement() }.get() }, value);
    if (result.hasException())
        return result;

    commitChange();
    return result;
}

ExceptionOr<void> SVGLength::convertToSpecifiedUnits(unsigned short unitType)
{
    if (isReadOnly())
        return Exception { ExceptionCode::NoModificationAllowedError };

    if (unitType == SVG_LENGTHTYPE_UNKNOWN || unitType > SVG_LENGTHTYPE_PC)
        return Exception { ExceptionCode::NotSupportedError };

    auto result = m_value.convertToSpecifiedUnits(SVGLengthContext { RefPtr { contextElement() }.get() }, static_cast<SVGLengthType>(unitType));
    if (result.hasException())
        return result;

    commitChange();
    return result;
}

} // namespace WebCore
