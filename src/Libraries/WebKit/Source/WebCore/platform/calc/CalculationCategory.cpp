/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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
#include "CalculationCategory.h"

#include <wtf/text/TextStream.h>

namespace WebCore {
namespace Calculation {

TextStream& operator<<(TextStream& ts, Category category)
{
    switch (category) {
    case Category::Integer: ts << "integer"; break;
    case Category::Number: ts << "number"; break;
    case Category::Percentage: ts << "percentage"; break;
    case Category::Length: ts << "length"; break;
    case Category::Angle: ts << "angle"; break;
    case Category::AnglePercentage: ts << "angle-percentage"; break;
    case Category::Time: ts << "time"; break;
    case Category::Frequency: ts << "frequency"; break;
    case Category::Resolution: ts << "resolution"; break;
    case Category::Flex: ts << "flex"; break;
    case Category::LengthPercentage: ts << "length-percentage"; break;
    }

    return ts;
}

} // namespace Calculation
} // namespace WebCore
