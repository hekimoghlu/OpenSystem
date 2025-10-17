/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
// class conv_transform
//
//----------------------------------------------------------------------------
#ifndef AGG_CONV_TRANSFORM_INCLUDED
#define AGG_CONV_TRANSFORM_INCLUDED

#include "agg_basics.h"
#include "agg_trans_affine.h"

namespace agg
{

    //----------------------------------------------------------conv_transform
    template<class VertexSource, class Transformer=trans_affine> class conv_transform
    {
    public:
        conv_transform(VertexSource& source, const Transformer& tr) :
            m_source(&source), m_trans(&tr) {}
        void attach(VertexSource& source) { m_source = &source; }

        void rewind(unsigned path_id) 
        { 
            m_source->rewind(path_id); 
        }

        unsigned vertex(double* x, double* y)
        {
            unsigned cmd = m_source->vertex(x, y);
            if(is_vertex(cmd))
            {
                m_trans->transform(x, y);
            }
            return cmd;
        }

        void transformer(const Transformer& tr)
        {
            m_trans = &tr;
        }

    private:
        conv_transform(const conv_transform<VertexSource>&);
        const conv_transform<VertexSource>& 
            operator = (const conv_transform<VertexSource>&);

        VertexSource*      m_source;
        const Transformer* m_trans;
    };


}

#endif
