/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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

#ifndef AGG_CONV_MARKER_ADAPTOR_INCLUDED
#define AGG_CONV_MARKER_ADAPTOR_INCLUDED

#include "agg_basics.h"
#include "agg_conv_adaptor_vcgen.h"
#include "agg_vcgen_vertex_sequence.h"

namespace agg
{

    //=====================================================conv_marker_adaptor
    template<class VertexSource, class Markers=null_markers>
    struct conv_marker_adaptor : 
    public conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence, Markers>
    {
        typedef Markers marker_type;
        typedef conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence, Markers> base_type;

        conv_marker_adaptor(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence, Markers>(vs)
        {
        }

        void shorten(double s) { base_type::generator().shorten(s); }
        double shorten() const { return base_type::generator().shorten(); }

    private:
        conv_marker_adaptor(const conv_marker_adaptor<VertexSource, Markers>&);
        const conv_marker_adaptor<VertexSource, Markers>& 
            operator = (const conv_marker_adaptor<VertexSource, Markers>&);
    };


}

#endif
