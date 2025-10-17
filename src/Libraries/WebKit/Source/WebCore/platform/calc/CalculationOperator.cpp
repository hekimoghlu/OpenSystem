/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
#include "CalculationOperator.h"

#include <wtf/text/TextStream.h>

namespace WebCore {
namespace Calculation {

TextStream& operator<<(TextStream& ts, Operator op)
{
    switch (op) {
    case Operator::Sum: ts << "+"; break;
    case Operator::Negate: ts << "-"; break;
    case Operator::Product: ts << "*"; break;
    case Operator::Invert: ts << "/"; break;
    case Operator::Min: ts << "min"; break;
    case Operator::Max: ts << "max"; break;
    case Operator::Clamp: ts << "clamp"; break;
    case Operator::Pow: ts << "pow"; break;
    case Operator::Sqrt: ts << "sqrt"; break;
    case Operator::Hypot: ts << "hypot"; break;
    case Operator::Sin: ts << "sin"; break;
    case Operator::Cos: ts << "cos"; break;
    case Operator::Tan: ts << "tan"; break;
    case Operator::Exp: ts << "exp"; break;
    case Operator::Log: ts << "log"; break;
    case Operator::Asin: ts << "asin"; break;
    case Operator::Acos: ts << "acos"; break;
    case Operator::Atan: ts << "atan"; break;
    case Operator::Atan2: ts << "atan2"; break;
    case Operator::Abs: ts << "abs"; break;
    case Operator::Sign: ts << "sign"; break;
    case Operator::Mod: ts << "mod"; break;
    case Operator::Rem: ts << "rem"; break;
    case Operator::Round: ts << "round"; break;
    case Operator::Up: ts << "up"; break;
    case Operator::Down: ts << "down"; break;
    case Operator::ToZero: ts << "to-zero"; break;
    case Operator::Nearest: ts << "nearest"; break;
    case Operator::Progress: ts << "progress"; break;
    case Operator::Random: ts << "random"; break;
    case Operator::Blend: ts << "blend"; break;
    }
    return ts;
}

} // namespace Calculation
} // namespace WebCore
