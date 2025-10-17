/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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

#ifndef AGG_VCGEN_SMOOTH_POLY1_INCLUDED
#define AGG_VCGEN_SMOOTH_POLY1_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"


namespace agg
{

    //======================================================vcgen_smooth_poly1
    //
    // See Implementation agg_vcgen_smooth_poly1.cpp 
    // Smooth polygon generator
    //
    //------------------------------------------------------------------------
    class vcgen_smooth_poly1
    {
        enum status_e
        {
            initial,
            ready,
            polygon,
            ctrl_b,
            ctrl_e,
            ctrl1,
            ctrl2,
            end_poly,
            stop
        };

    public:
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;

        vcgen_smooth_poly1();

        void   smooth_value(double v) { m_smooth_value = v * 0.5; }
        double smooth_value() const { return m_smooth_value * 2.0; }

        // Vertex Generator Interface
        void remove_all();
        void add_vertex(double x, double y, unsigned cmd);

        // Vertex Source Interface
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        vcgen_smooth_poly1(const vcgen_smooth_poly1&);
        const vcgen_smooth_poly1& operator = (const vcgen_smooth_poly1&);

        void calculate(const vertex_dist& v0, 
                       const vertex_dist& v1, 
                       const vertex_dist& v2,
                       const vertex_dist& v3);

        vertex_storage m_src_vertices;
        double         m_smooth_value;
        unsigned       m_closed;
        status_e       m_status;
        unsigned       m_src_vertex;
        double         m_ctrl1_x;
        double         m_ctrl1_y;
        double         m_ctrl2_x;
        double         m_ctrl2_y;
    };

}


#endif

