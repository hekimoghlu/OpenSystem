/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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
// conv_marker
//
//----------------------------------------------------------------------------
#ifndef AGG_CONV_MARKER_INCLUDED
#define AGG_CONV_MARKER_INCLUDED

#include "agg_basics.h"
#include "agg_trans_affine.h"

namespace agg
{
    //-------------------------------------------------------------conv_marker
    template<class MarkerLocator, class MarkerShapes>
    class conv_marker
    {
    public:
        conv_marker(MarkerLocator& ml, MarkerShapes& ms);

        trans_affine& transform() { return m_transform; }
        const trans_affine& transform() const { return m_transform; }

        void rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        conv_marker(const conv_marker<MarkerLocator, MarkerShapes>&);
        const conv_marker<MarkerLocator, MarkerShapes>& 
            operator = (const conv_marker<MarkerLocator, MarkerShapes>&);

        enum status_e 
        {
            initial,
            markers,
            polygon,
            stop
        };

        MarkerLocator* m_marker_locator;
        MarkerShapes*  m_marker_shapes;
        trans_affine   m_transform;
        trans_affine   m_mtx;
        status_e       m_status;
        unsigned       m_marker;
        unsigned       m_num_markers;
    };


    //------------------------------------------------------------------------
    template<class MarkerLocator, class MarkerShapes> 
    conv_marker<MarkerLocator, MarkerShapes>::conv_marker(MarkerLocator& ml, MarkerShapes& ms) :
        m_marker_locator(&ml),
        m_marker_shapes(&ms),
        m_status(initial),
        m_marker(0),
        m_num_markers(1)
    {
    }


    //------------------------------------------------------------------------
    template<class MarkerLocator, class MarkerShapes> 
    void conv_marker<MarkerLocator, MarkerShapes>::rewind(unsigned)
    {
        m_status = initial;
        m_marker = 0;
        m_num_markers = 1;
    }


    //------------------------------------------------------------------------
    template<class MarkerLocator, class MarkerShapes> 
    unsigned conv_marker<MarkerLocator, MarkerShapes>::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_move_to;
        double x1, y1, x2, y2;

        while(!is_stop(cmd))
        {
            switch(m_status)
            {
            case initial:
                if(m_num_markers == 0)
                {
                   cmd = path_cmd_stop;
                   break;
                }
                m_marker_locator->rewind(m_marker);
                ++m_marker;
                m_num_markers = 0;
                m_status = markers;

            case markers:
                if(is_stop(m_marker_locator->vertex(&x1, &y1)))
                {
                    m_status = initial;
                    break;
                }
                if(is_stop(m_marker_locator->vertex(&x2, &y2)))
                {
                    m_status = initial;
                    break;
                }
                ++m_num_markers;
                m_mtx = m_transform;
                m_mtx *= trans_affine_rotation(atan2(y2 - y1, x2 - x1));
                m_mtx *= trans_affine_translation(x1, y1);
                m_marker_shapes->rewind(m_marker - 1);
                m_status = polygon;

            case polygon:
                cmd = m_marker_shapes->vertex(x, y);
                if(is_stop(cmd))
                {
                    cmd = path_cmd_move_to;
                    m_status = markers;
                    break;
                }
                m_mtx.transform(x, y);
                return cmd;

            case stop:
                cmd = path_cmd_stop;
                break;
            }
        }
        return cmd;
    }

}


#endif

