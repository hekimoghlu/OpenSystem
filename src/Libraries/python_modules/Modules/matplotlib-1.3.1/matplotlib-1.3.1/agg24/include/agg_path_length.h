/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#ifndef AGG_PATH_LENGTH_INCLUDED
#define AGG_PATH_LENGTH_INCLUDED

#include "agg_math.h"

namespace agg
{
    template<class VertexSource> 
    double path_length(VertexSource& vs, unsigned path_id = 0)
    {
        double len = 0.0;
        double start_x = 0.0;
        double start_y = 0.0;
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 0.0;
        double y2 = 0.0;
        bool first = true;

        unsigned cmd;
        vs.rewind(path_id);
        while(!is_stop(cmd = vs.vertex(&x2, &y2)))
        {
            if(is_vertex(cmd))
            {
                if(first || is_move_to(cmd))
                {
                    start_x = x2;
                    start_y = y2;
                }
                else
                {
                    len += calc_distance(x1, y1, x2, y2);
                }
                x1 = x2;
                y1 = y2;
                first = false;
            }
            else
            {
                if(is_close(cmd) && !first)
                {
                    len += calc_distance(x1, y1, start_x, start_y);
                }
            }
        }
        return len;
    }
}

#endif
