/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
// Horizontal span interpolator for use with an arbitrary transformer
// The efficiency highly depends on the operations done in the transformer
//
//----------------------------------------------------------------------------

#ifndef AGG_SPAN_INTERPOLATOR_TRANS_INCLUDED
#define AGG_SPAN_INTERPOLATOR_TRANS_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //=================================================span_interpolator_trans
    template<class Transformer, unsigned SubpixelShift = 8> 
    class span_interpolator_trans
    {
    public:
        typedef Transformer trans_type;
        enum subpixel_scale_e
        {
            subpixel_shift = SubpixelShift,
            subpixel_scale = 1 << subpixel_shift
        };

        //--------------------------------------------------------------------
        span_interpolator_trans() {}
        span_interpolator_trans(const trans_type& trans) : m_trans(&trans) {}
        span_interpolator_trans(const trans_type& trans,
                                double x, double y, unsigned) :
            m_trans(&trans)
        {
            begin(x, y, 0);
        }

        //----------------------------------------------------------------
        const trans_type& transformer() const { return *m_trans; }
        void transformer(const trans_type& trans) { m_trans = &trans; }

        //----------------------------------------------------------------
        void begin(double x, double y, unsigned)
        {
            m_x = x;
            m_y = y;
            m_trans->transform(&x, &y);
            m_ix = iround(x * subpixel_scale);
            m_iy = iround(y * subpixel_scale);
        }

        //----------------------------------------------------------------
        void operator++()
        {
            m_x += 1.0;
            double x = m_x;
            double y = m_y;
            m_trans->transform(&x, &y);
            m_ix = iround(x * subpixel_scale);
            m_iy = iround(y * subpixel_scale);
        }

        //----------------------------------------------------------------
        void coordinates(int* x, int* y) const
        {
            *x = m_ix;
            *y = m_iy;
        }

    private:
        const trans_type* m_trans;
        double            m_x;
        double            m_y;
        int               m_ix;
        int               m_iy;
    };

}

#endif
