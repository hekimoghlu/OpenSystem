/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#ifndef AGG_CONV_BSPLINE_INCLUDED
#define AGG_CONV_BSPLINE_INCLUDED

#include "agg_basics.h"
#include "agg_vcgen_bspline.h"
#include "agg_conv_adaptor_vcgen.h"


namespace agg
{

    //---------------------------------------------------------conv_bspline
    template<class VertexSource> 
    struct conv_bspline : public conv_adaptor_vcgen<VertexSource, vcgen_bspline>
    {
        typedef conv_adaptor_vcgen<VertexSource, vcgen_bspline> base_type;

        conv_bspline(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_bspline>(vs) {}

        void   interpolation_step(double v) { base_type::generator().interpolation_step(v); }
        double interpolation_step() const { return base_type::generator().interpolation_step(); }

    private:
        conv_bspline(const conv_bspline<VertexSource>&);
        const conv_bspline<VertexSource>& 
            operator = (const conv_bspline<VertexSource>&);
    };

}


#endif

