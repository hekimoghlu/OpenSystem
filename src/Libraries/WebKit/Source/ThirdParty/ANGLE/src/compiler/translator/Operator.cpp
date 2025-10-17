/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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

//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/Operator_autogen.h"

#include "common/debug.h"

namespace sh
{

const char *GetOperatorString(TOperator op)
{
    switch (op)
    {
            // Note: EOpNull and EOpCall* can't be handled here.

        case EOpNegative:
            return "-";
        case EOpPositive:
            return "+";
        case EOpLogicalNot:
            return "!";
        case EOpBitwiseNot:
            return "~";

        case EOpPostIncrement:
            return "++";
        case EOpPostDecrement:
            return "--";
        case EOpPreIncrement:
            return "++";
        case EOpPreDecrement:
            return "--";

        case EOpArrayLength:
            return ".length()";

        case EOpAdd:
            return "+";
        case EOpSub:
            return "-";
        case EOpMul:
            return "*";
        case EOpDiv:
            return "/";
        case EOpIMod:
            return "%";

        case EOpEqual:
            return "==";
        case EOpNotEqual:
            return "!=";
        case EOpLessThan:
            return "<";
        case EOpGreaterThan:
            return ">";
        case EOpLessThanEqual:
            return "<=";
        case EOpGreaterThanEqual:
            return ">=";

        case EOpComma:
            return ",";

        // Fall-through.
        case EOpVectorTimesScalar:
        case EOpVectorTimesMatrix:
        case EOpMatrixTimesVector:
        case EOpMatrixTimesScalar:
        case EOpMatrixTimesMatrix:
            return "*";

        case EOpLogicalOr:
            return "||";
        case EOpLogicalXor:
            return "^^";
        case EOpLogicalAnd:
            return "&&";

        case EOpBitShiftLeft:
            return "<<";
        case EOpBitShiftRight:
            return ">>";

        case EOpBitwiseAnd:
            return "&";
        case EOpBitwiseXor:
            return "^";
        case EOpBitwiseOr:
            return "|";

        // Fall-through.
        case EOpIndexDirect:
        case EOpIndexIndirect:
            return "[]";

        case EOpIndexDirectStruct:
        case EOpIndexDirectInterfaceBlock:
            return ".";

        case EOpAssign:
        case EOpInitialize:
            return "=";
        case EOpAddAssign:
            return "+=";
        case EOpSubAssign:
            return "-=";

        // Fall-through.
        case EOpMulAssign:
        case EOpVectorTimesMatrixAssign:
        case EOpVectorTimesScalarAssign:
        case EOpMatrixTimesScalarAssign:
        case EOpMatrixTimesMatrixAssign:
            return "*=";

        case EOpDivAssign:
            return "/=";
        case EOpIModAssign:
            return "%=";
        case EOpBitShiftLeftAssign:
            return "<<=";
        case EOpBitShiftRightAssign:
            return ">>=";
        case EOpBitwiseAndAssign:
            return "&=";
        case EOpBitwiseXorAssign:
            return "^=";
        case EOpBitwiseOrAssign:
            return "|=";

        default:
            UNREACHABLE();
            break;
    }
    return "";
}

bool IsAssignment(TOperator op)
{
    switch (op)
    {
        case EOpPostIncrement:
        case EOpPostDecrement:
        case EOpPreIncrement:
        case EOpPreDecrement:
        case EOpAssign:
        case EOpAddAssign:
        case EOpSubAssign:
        case EOpMulAssign:
        case EOpVectorTimesMatrixAssign:
        case EOpVectorTimesScalarAssign:
        case EOpMatrixTimesScalarAssign:
        case EOpMatrixTimesMatrixAssign:
        case EOpDivAssign:
        case EOpIModAssign:
        case EOpBitShiftLeftAssign:
        case EOpBitShiftRightAssign:
        case EOpBitwiseAndAssign:
        case EOpBitwiseXorAssign:
        case EOpBitwiseOrAssign:
            return true;
        default:
            return false;
    }
}

}  // namespace sh
