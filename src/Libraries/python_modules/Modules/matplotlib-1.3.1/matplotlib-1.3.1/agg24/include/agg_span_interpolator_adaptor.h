/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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

#ifndef AGG_SPAN_INTERPOLATOR_ADAPTOR_INCLUDED
#define AGG_SPAN_INTERPOLATOR_ADAPTOR_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //===============================================span_interpolator_adaptor
    template<class Interpolator, class Distortion>
    class span_interpolator_adaptor : public Interpolator
    {
    public:
        typedef Interpolator base_type;
        typedef typename base_type::trans_type trans_type;
        typedef Distortion distortion_type;

        //--------------------------------------------------------------------
        span_interpolator_adaptor() {}
        span_interpolator_adaptor(const trans_type& trans, 
                                  const distortion_type& dist) :
            base_type(trans),
            m_distortion(&dist)
        {   
        }

        //--------------------------------------------------------------------
        span_interpolator_adaptor(const trans_type& trans,
                                  const distortion_type& dist,
                                  double x, double y, unsigned len) :
            base_type(trans, x, y, len),
            m_distortion(&dist)
        {
        }

        //--------------------------------------------------------------------
        const distortion_type& distortion() const
        {
            return *m_distortion;
        }

        //--------------------------------------------------------------------
        void distortion(const distortion_type& dist)
        {
            m_distortion = dist;
        }

        //--------------------------------------------------------------------
        void coordinates(int* x, int* y) const
        {
            base_type::coordinates(x, y);
            m_distortion->calculate(x, y);
        }

    private:
        //--------------------------------------------------------------------
        const distortion_type* m_distortion;
    };
}


#endif
