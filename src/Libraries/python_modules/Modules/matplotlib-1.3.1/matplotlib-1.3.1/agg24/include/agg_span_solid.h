/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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
// span_solid_rgba8
//
//----------------------------------------------------------------------------

#ifndef AGG_SPAN_SOLID_INCLUDED
#define AGG_SPAN_SOLID_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //--------------------------------------------------------------span_solid
    template<class ColorT> class span_solid
    {
    public:
        typedef ColorT color_type;

        //--------------------------------------------------------------------
        void color(const color_type& c) { m_color = c; }
        const color_type& color() const { return m_color; }

        //--------------------------------------------------------------------
        void prepare() {}

        //--------------------------------------------------------------------
        void generate(color_type* span, int x, int y, unsigned len)
        {   
            do { *span++ = m_color; } while(--len);
        }

    private:
        color_type m_color;
    };


}

#endif
