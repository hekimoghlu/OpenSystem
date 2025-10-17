/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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

#include "bzfstream.h"
#include "optuple.h"

int main()
{
    std::cout << "Generating <vecwhere.cc>" << std::endl;

    bzofstream ofs("../vecwhere.cc", "where(X,Y,Z) function for vectors",
        __FILE__, "BZ_VECWHERE_CC");

    ofs.beginNamespace();

    OperandTuple ops(3);

    do {
        if (ops[0].isScalar())
            continue;

        if (ops[1].isScalar() && ops[2].isScalar())
        {
            if (ops.operandIndex(1) != ops.operandIndex(2))
                continue;
        }

        ofs << "// where(";
        ops.printTypes(ofs);
        ofs << ")" << std::endl;

        int complexFlag = 0;
        if (ops.anyComplex())
        {
            ofs << "#ifdef BZ_HAVE_COMPLEX" << std::endl;
            complexFlag = 1;
        }

        ops.printTemplates(ofs);
        ofs << std::endl << "inline" << std::endl;
        ofs << "_bz_VecExpr<_bz_VecWhere<";
        ops.printIterators(ofs, 1);
        ofs << " > >" << std::endl;

        ofs << "where(";
        ops.printArgumentList(ofs, 1);
        ofs << ")" << std::endl
            << "{ " << std::endl;

        ofs << "    typedef _bz_VecWhere<";
        ops.printIterators(ofs, 1);
        ofs << " > T_expr;" << std::endl << std::endl;

        ofs << "    return _bz_VecExpr<T_expr>(T_expr(";
        ops.printInitializationList(ofs, 1);
        ofs << "));" << std::endl
            << "}" << std::endl;
      
        if (complexFlag)
            ofs << "#endif // BZ_HAVE_COMPLEX" << std::endl;

        ofs << std::endl;
    } while (++ops);

    std::cout << ops.numSpecializations() << " specializations written." << std::endl;
}

