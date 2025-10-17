/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#ifndef BZ_TINYMATIO_CC
#define BZ_TINYMATIO_CC

#ifndef BZ_TINYMAT_H
 #include <blitz/tinymat.h>
#endif

BZ_NAMESPACE(blitz)

template <typename P_numtype, int N_rows, int N_columns>
ostream& operator<<(ostream& os,
    const TinyMatrix<P_numtype, N_rows, N_columns>& x)
{
    os << "(" << N_rows << "," << N_columns << "): " << endl;
    for (int i=0; i < N_rows; ++i)
    {
        os << " [ ";
        for (int j=0; j < N_columns; ++j)
        {
            os << setw(10) << x(i,j);
            if (!((j+1)%7))
                os << endl << "  ";
        }
        os << " ]" << endl;
    }
    return os;
}

template <typename P_numtype, int N_rows, int N_columns>
istream& operator>>(istream& is, 
    TinyMatrix<P_numtype, N_rows, N_columns>& x)
{
    int rows, columns;
    char sep;
             
    is >> rows >> columns;

    BZPRECHECK(rows == N_rows, "Size mismatch in number of rows");
    BZPRECHECK(columns == N_columns, "Size mismatch in number of columns");

    for (int i=0; i < N_rows; ++i) 
    {
        is >> sep;
        BZPRECHECK(sep == '[', "Format error while scanning input matrix"
            << endl << " (expected '[' before beginning of row data)");
        for (int j = 0; j < N_columns; ++j)
        {
            BZPRECHECK(!is.bad(), "Premature end of input while scanning matrix");
            is >> x(i,j);
        }
        is >> sep;
        BZPRECHECK(sep == ']', "Format error while scanning input matrix"
            << endl << " (expected ']' after end of row data)");
    }

    return is;
}

BZ_NAMESPACE_END

#endif // BZ_TINYMATIO_CC

