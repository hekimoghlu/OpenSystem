/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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

//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// Simple arrowhead/arrowtail generator 
//
//----------------------------------------------------------------------------
#ifndef AGG_ARROWHEAD_INCLUDED
#define AGG_ARROWHEAD_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //===============================================================arrowhead
    //
    // See implementation agg_arrowhead.cpp 
    //
    class arrowhead
    {
    public:
        arrowhead();

        void head(double d1, double d2, double d3, double d4)
        {
            m_head_d1 = d1;
            m_head_d2 = d2;
            m_head_d3 = d3;
            m_head_d4 = d4;
            m_head_flag = true;
        }

        void head()    { m_head_flag = true; }
        void no_head() { m_head_flag = false; }

        void tail(double d1, double d2, double d3, double d4)
        {
            m_tail_d1 = d1;
            m_tail_d2 = d2;
            m_tail_d3 = d3;
            m_tail_d4 = d4;
            m_tail_flag = true;
        }

        void tail()    { m_tail_flag = true;  }
        void no_tail() { m_tail_flag = false; }

        void rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        double   m_head_d1;
        double   m_head_d2;
        double   m_head_d3;
        double   m_head_d4;
        double   m_tail_d1;
        double   m_tail_d2;
        double   m_tail_d3;
        double   m_tail_d4;
        bool     m_head_flag;
        bool     m_tail_flag;
        double   m_coord[16];
        unsigned m_cmd[8];
        unsigned m_curr_id;
        unsigned m_curr_coord;
    };

}

#endif
