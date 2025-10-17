/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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

#include <math.h>
#include "agg_trans_warp_magnifier.h"

namespace agg
{

    //------------------------------------------------------------------------
    void trans_warp_magnifier::transform(double* x, double* y) const
    {
        double dx = *x - m_xc;
        double dy = *y - m_yc;
        double r = sqrt(dx * dx + dy * dy);
        if(r < m_radius)
        {
            *x = m_xc + dx * m_magn;
            *y = m_yc + dy * m_magn;
            return;
        }

        double m = (r + m_radius * (m_magn - 1.0)) / r;
        *x = m_xc + dx * m;
        *y = m_yc + dy * m;
    }

    //------------------------------------------------------------------------
    void trans_warp_magnifier::inverse_transform(double* x, double* y) const
    {
        // New version by Andrew Skalkin
        //-----------------
        double dx = *x - m_xc;
        double dy = *y - m_yc;
        double r = sqrt(dx * dx + dy * dy);

        if(r < m_radius * m_magn) 
        {
            *x = m_xc + dx / m_magn;
            *y = m_yc + dy / m_magn;
        }
        else
        {
            double rnew = r - m_radius * (m_magn - 1.0);
            *x = m_xc + rnew * dx / r; 
            *y = m_yc + rnew * dy / r;
        }

        // Old version
        //-----------------
        //trans_warp_magnifier t(*this);
        //t.magnification(1.0 / m_magn);
        //t.radius(m_radius * m_magn);
        //t.transform(x, y);
    }


}
