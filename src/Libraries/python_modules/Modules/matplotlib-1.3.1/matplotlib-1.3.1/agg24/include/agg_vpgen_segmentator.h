/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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

#ifndef AGG_VPGEN_SEGMENTATOR_INCLUDED
#define AGG_VPGEN_SEGMENTATOR_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{

    //=======================================================vpgen_segmentator
    // 
    // See Implementation agg_vpgen_segmentator.cpp
    //
    class vpgen_segmentator
    {
    public:
        vpgen_segmentator() : m_approximation_scale(1.0) {}

        void approximation_scale(double s) { m_approximation_scale = s;     }
        double approximation_scale() const { return m_approximation_scale;  }

        static bool auto_close()   { return false; }
        static bool auto_unclose() { return false; }

        void reset() { m_cmd = path_cmd_stop; }
        void move_to(double x, double y);
        void line_to(double x, double y);
        unsigned vertex(double* x, double* y);

    private:
        double   m_approximation_scale;
        double   m_x1;
        double   m_y1;
        double   m_dx;
        double   m_dy;
        double   m_dl;
        double   m_ddl;
        unsigned m_cmd;
    };



}

#endif

