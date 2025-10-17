/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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

#include "agg_vpgen_clip_polyline.h"
#include "agg_clip_liang_barsky.h"

namespace agg
{
    //----------------------------------------------------------------------------
    void vpgen_clip_polyline::reset()
    {
        m_vertex = 0;
        m_num_vertices = 0;
        m_move_to = false;
    }

    //----------------------------------------------------------------------------
    void vpgen_clip_polyline::move_to(double x, double y)
    {
        m_vertex = 0;
        m_num_vertices = 0;
        m_x1 = x;
        m_y1 = y;
        m_move_to = true;
    }

    //----------------------------------------------------------------------------
    void vpgen_clip_polyline::line_to(double x, double y)
    {
        double x2 = x;
        double y2 = y;
        unsigned flags = clip_line_segment(&m_x1, &m_y1, &x2, &y2, m_clip_box);

        m_vertex = 0;
        m_num_vertices = 0;
        if((flags & 4) == 0)
        {
            if((flags & 1) != 0 || m_move_to)
            {
                m_x[0] = m_x1;
                m_y[0] = m_y1;
                m_cmd[0] = path_cmd_move_to;
                m_num_vertices = 1;
            }
            m_x[m_num_vertices] = x2;
            m_y[m_num_vertices] = y2;
            m_cmd[m_num_vertices++] = path_cmd_line_to;
            m_move_to = (flags & 2) != 0;
        }
        m_x1 = x;
        m_y1 = y;
    }

    //----------------------------------------------------------------------------
    unsigned vpgen_clip_polyline::vertex(double* x, double* y)
    {
        if(m_vertex < m_num_vertices)
        {
            *x = m_x[m_vertex];
            *y = m_y[m_vertex];
            return m_cmd[m_vertex++];
        }
        return path_cmd_stop;
    }
}
