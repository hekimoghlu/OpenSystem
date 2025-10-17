/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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

#ifndef AGG_CONV_UNCLOSE_POLYGON_INCLUDED
#define AGG_CONV_UNCLOSE_POLYGON_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //====================================================conv_unclose_polygon
    template<class VertexSource> class conv_unclose_polygon
    {
    public:
        explicit conv_unclose_polygon(VertexSource& vs) : m_source(&vs) {}
        void attach(VertexSource& source) { m_source = &source; }

        void rewind(unsigned path_id)
        {
            m_source->rewind(path_id);
        }

        unsigned vertex(double* x, double* y)
        {
            unsigned cmd = m_source->vertex(x, y);
            if(is_end_poly(cmd)) cmd &= ~path_flags_close;
            return cmd;
        }

    private:
        conv_unclose_polygon(const conv_unclose_polygon<VertexSource>&);
        const conv_unclose_polygon<VertexSource>& 
            operator = (const conv_unclose_polygon<VertexSource>&);

        VertexSource* m_source;
    };

}

#endif
