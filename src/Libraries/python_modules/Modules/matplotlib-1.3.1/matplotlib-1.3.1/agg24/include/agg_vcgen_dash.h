/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
// Line dash generator
//
//----------------------------------------------------------------------------
#ifndef AGG_VCGEN_DASH_INCLUDED
#define AGG_VCGEN_DASH_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    //---------------------------------------------------------------vcgen_dash
    //
    // See Implementation agg_vcgen_dash.cpp
    //
    class vcgen_dash
    {
        enum max_dashes_e
        {
            max_dashes = 32
        };

        enum status_e
        {
            initial,
            ready,
            polyline,
            stop
        };

    public:
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;

        vcgen_dash();

        void remove_all_dashes();
        void add_dash(double dash_len, double gap_len);
        void dash_start(double ds);

        void shorten(double s) { m_shorten = s; }
        double shorten() const { return m_shorten; }

        // Vertex Generator Interface
        void remove_all();
        void add_vertex(double x, double y, unsigned cmd);

        // Vertex Source Interface
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        vcgen_dash(const vcgen_dash&);
        const vcgen_dash& operator = (const vcgen_dash&);

        void calc_dash_start(double ds);

        double             m_dashes[max_dashes];
        double             m_total_dash_len;
        unsigned           m_num_dashes;
        double             m_dash_start;
        double             m_shorten;
        double             m_curr_dash_start;
        unsigned           m_curr_dash;
        double             m_curr_rest;
        const vertex_dist* m_v1;
        const vertex_dist* m_v2;

        vertex_storage m_src_vertices;
        unsigned       m_closed;
        status_e       m_status;
        unsigned       m_src_vertex;
    };


}

#endif
