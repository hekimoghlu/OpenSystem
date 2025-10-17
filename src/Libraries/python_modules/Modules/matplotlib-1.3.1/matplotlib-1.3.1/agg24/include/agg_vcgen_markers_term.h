/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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

#ifndef AGG_VCGEN_MARKERS_TERM_INCLUDED
#define AGG_VCGEN_MARKERS_TERM_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    //======================================================vcgen_markers_term
    //
    // See Implemantation agg_vcgen_markers_term.cpp
    // Terminal markers generator (arrowhead/arrowtail)
    //
    //------------------------------------------------------------------------
    class vcgen_markers_term
    {
    public:
        vcgen_markers_term() : m_curr_id(0), m_curr_idx(0) {}

        // Vertex Generator Interface
        void remove_all();
        void add_vertex(double x, double y, unsigned cmd);

        // Vertex Source Interface
        void rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        vcgen_markers_term(const vcgen_markers_term&);
        const vcgen_markers_term& operator = (const vcgen_markers_term&);

        struct coord_type
        {
            double x, y;

            coord_type() {}
            coord_type(double x_, double y_) : x(x_), y(y_) {}
        };

        typedef pod_bvector<coord_type, 6> coord_storage; 

        coord_storage m_markers;
        unsigned      m_curr_id;
        unsigned      m_curr_idx;
    };


}

#endif
