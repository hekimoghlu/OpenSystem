/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
// Terminal markers generator (arrowhead/arrowtail)
//
//----------------------------------------------------------------------------

#include "agg_vcgen_markers_term.h"

namespace agg
{

    //------------------------------------------------------------------------
    void vcgen_markers_term::remove_all()
    {
        m_markers.remove_all();
    }


    //------------------------------------------------------------------------
    void vcgen_markers_term::add_vertex(double x, double y, unsigned cmd)
    {
        if(is_move_to(cmd))
        {
            if(m_markers.size() & 1)
            {
                // Initial state, the first coordinate was added.
                // If two of more calls of start_vertex() occures
                // we just modify the last one.
                m_markers.modify_last(coord_type(x, y));
            }
            else
            {
                m_markers.add(coord_type(x, y));
            }
        }
        else
        {
            if(is_vertex(cmd))
            {
                if(m_markers.size() & 1)
                {
                    // Initial state, the first coordinate was added.
                    // Add three more points, 0,1,1,0
                    m_markers.add(coord_type(x, y));
                    m_markers.add(m_markers[m_markers.size() - 1]);
                    m_markers.add(m_markers[m_markers.size() - 3]);
                }
                else
                {
                    if(m_markers.size())
                    {
                        // Replace two last points: 0,1,1,0 -> 0,1,2,1
                        m_markers[m_markers.size() - 1] = m_markers[m_markers.size() - 2];
                        m_markers[m_markers.size() - 2] = coord_type(x, y);
                    }
                }
            }
        }
    }


    //------------------------------------------------------------------------
    void vcgen_markers_term::rewind(unsigned path_id)
    {
        m_curr_id = path_id * 2;
        m_curr_idx = m_curr_id;
    }


    //------------------------------------------------------------------------
    unsigned vcgen_markers_term::vertex(double* x, double* y)
    {
        if(m_curr_id > 2 || m_curr_idx >= m_markers.size()) 
        {
            return path_cmd_stop;
        }
        const coord_type& c = m_markers[m_curr_idx];
        *x = c.x;
        *y = c.y;
        if(m_curr_idx & 1)
        {
            m_curr_idx += 3;
            return path_cmd_line_to;
        }
        ++m_curr_idx;
        return path_cmd_move_to;
    }


}
