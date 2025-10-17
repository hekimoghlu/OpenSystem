/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#include "agg_vpgen_segmentator.h"

namespace agg
{

    void vpgen_segmentator::move_to(double x, double y)
    {
        m_x1 = x;
        m_y1 = y;
        m_dx = 0.0;
        m_dy = 0.0;
        m_dl = 2.0;
        m_ddl = 2.0;
        m_cmd = path_cmd_move_to;
    }

    void vpgen_segmentator::line_to(double x, double y)
    {
        m_x1 += m_dx;
        m_y1 += m_dy;
        m_dx  = x - m_x1;
        m_dy  = y - m_y1;
        double len = sqrt(m_dx * m_dx + m_dy * m_dy) * m_approximation_scale;
        if(len < 1e-30) len = 1e-30;
        m_ddl = 1.0 / len;
        m_dl  = (m_cmd == path_cmd_move_to) ? 0.0 : m_ddl;
        if(m_cmd == path_cmd_stop) m_cmd = path_cmd_line_to;
    }

    unsigned vpgen_segmentator::vertex(double* x, double* y)
    {
        if(m_cmd == path_cmd_stop) return path_cmd_stop;

        unsigned cmd = m_cmd;
        m_cmd = path_cmd_line_to;
        if(m_dl >= 1.0 - m_ddl)
        {
            m_dl = 1.0;
            m_cmd = path_cmd_stop;
            *x = m_x1 + m_dx;
            *y = m_y1 + m_dy;
            return cmd;
        }
        *x = m_x1 + m_dx * m_dl;
        *y = m_y1 + m_dy * m_dl;
        m_dl += m_ddl;
        return cmd;
    }

}

