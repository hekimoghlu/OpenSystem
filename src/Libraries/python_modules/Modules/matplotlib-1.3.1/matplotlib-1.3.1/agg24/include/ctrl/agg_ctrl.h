/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
// Function render_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_CTRL_INCLUDED
#define AGG_CTRL_INCLUDED

#include "agg_trans_affine.h"
#include "agg_renderer_scanline.h"

namespace agg
{

    //--------------------------------------------------------------------ctrl
    class ctrl
    {
    public:
        //--------------------------------------------------------------------
        virtual ~ctrl() {}
        ctrl(double x1, double y1, double x2, double y2, bool flip_y) :
            m_x1(x1), m_y1(y1), m_x2(x2), m_y2(y2), 
            m_flip_y(flip_y),
            m_mtx(0)
        {
        }

        //--------------------------------------------------------------------
        virtual bool in_rect(double x, double y) const = 0;
        virtual bool on_mouse_button_down(double x, double y) = 0;
        virtual bool on_mouse_button_up(double x, double y) = 0;
        virtual bool on_mouse_move(double x, double y, bool button_flag) = 0;
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up) = 0;

        //--------------------------------------------------------------------
        void transform(const trans_affine& mtx) { m_mtx = &mtx; }
        void no_transform() { m_mtx = 0; }

        //--------------------------------------------------------------------
        void transform_xy(double* x, double* y) const
        {
            if(m_flip_y) *y = m_y1 + m_y2 - *y;
            if(m_mtx) m_mtx->transform(x, y);
        }

        //--------------------------------------------------------------------
        void inverse_transform_xy(double* x, double* y) const
        {
            if(m_mtx) m_mtx->inverse_transform(x, y);
            if(m_flip_y) *y = m_y1 + m_y2 - *y;
        }

        //--------------------------------------------------------------------
        double scale() const { return m_mtx ? m_mtx->scale() : 1.0; }

    private:
        ctrl(const ctrl&);
        const ctrl& operator = (const ctrl&);

    protected:
        double m_x1;
        double m_y1;
        double m_x2;
        double m_y2;

    private:
        bool m_flip_y;
        const trans_affine* m_mtx;
    };


    //--------------------------------------------------------------------
    template<class Rasterizer, class Scanline, class Renderer, class Ctrl> 
    void render_ctrl(Rasterizer& ras, Scanline& sl, Renderer& r, Ctrl& c)
    {
        unsigned i;
        for(i = 0; i < c.num_paths(); i++)
        {
            ras.reset();
            ras.add_path(c, i);
            render_scanlines_aa_solid(ras, sl, r, c.color(i));
        }
    }


    //--------------------------------------------------------------------
    template<class Rasterizer, class Scanline, class Renderer, class Ctrl> 
    void render_ctrl_rs(Rasterizer& ras, Scanline& sl, Renderer& r, Ctrl& c)
    {
        unsigned i;
        for(i = 0; i < c.num_paths(); i++)
        {
            ras.reset();
            ras.add_path(c, i);
            r.color(c.color(i));
            render_scanlines(ras, sl, r);
        }
    }


}


#endif
