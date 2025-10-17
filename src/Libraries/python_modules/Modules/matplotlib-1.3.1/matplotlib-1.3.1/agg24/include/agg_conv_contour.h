/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
// conv_stroke
//
//----------------------------------------------------------------------------
#ifndef AGG_CONV_CONTOUR_INCLUDED
#define AGG_CONV_CONTOUR_INCLUDED

#include "agg_basics.h"
#include "agg_vcgen_contour.h"
#include "agg_conv_adaptor_vcgen.h"

namespace agg
{

    //-----------------------------------------------------------conv_contour
    template<class VertexSource> 
    struct conv_contour : public conv_adaptor_vcgen<VertexSource, vcgen_contour>
    {
        typedef conv_adaptor_vcgen<VertexSource, vcgen_contour> base_type;

        conv_contour(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_contour>(vs)
        {
        }

        void line_join(line_join_e lj) { base_type::generator().line_join(lj); }
        void inner_join(inner_join_e ij) { base_type::generator().inner_join(ij); }
        void width(double w) { base_type::generator().width(w); }
        void miter_limit(double ml) { base_type::generator().miter_limit(ml); }
        void miter_limit_theta(double t) { base_type::generator().miter_limit_theta(t); }
        void inner_miter_limit(double ml) { base_type::generator().inner_miter_limit(ml); }
        void approximation_scale(double as) { base_type::generator().approximation_scale(as); }
        void auto_detect_orientation(bool v) { base_type::generator().auto_detect_orientation(v); }

        line_join_e line_join() const { return base_type::generator().line_join(); }
        inner_join_e inner_join() const { return base_type::generator().inner_join(); }
        double width() const { return base_type::generator().width(); }
        double miter_limit() const { return base_type::generator().miter_limit(); }
        double inner_miter_limit() const { return base_type::generator().inner_miter_limit(); }
        double approximation_scale() const { return base_type::generator().approximation_scale(); }
        bool auto_detect_orientation() const { return base_type::generator().auto_detect_orientation(); }

    private:
        conv_contour(const conv_contour<VertexSource>&);
        const conv_contour<VertexSource>& 
            operator = (const conv_contour<VertexSource>&);
    };

}

#endif
