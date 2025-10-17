/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
// Copyright (C) 2002-2005 Maxim Shemanarev (McSeem)
// Copyright (C) 2002 Hansruedi Baer (MacOS support)
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
//			baer@karto.baug.eth.ch
//----------------------------------------------------------------------------
//
// class pixel_map
//
//----------------------------------------------------------------------------
#ifndef AGG_MAC_PMAP_INCLUDED
#define AGG_MAC_PMAP_INCLUDED


#include <stdio.h>
#include <Carbon.h>


namespace agg
{
    enum org_e
    {
        org_mono8   = 8,
        org_color16 = 16,
        org_color24 = 24,
        org_color32 = 32
    };

    class pixel_map
    {
    public:
        ~pixel_map();
        pixel_map();

    public:
        void        destroy();
        void        create(unsigned width, 
                           unsigned height, 
                           org_e    org,
                           unsigned clear_val=255);

        void        clear(unsigned clear_val=255);
        bool        load_from_qt(const char* filename);
        bool        save_as_qt(const char* filename) const;

        void        draw(WindowRef window, 
                         const Rect* device_rect=0, 
                         const Rect* bmp_rect=0) const;
        void        draw(WindowRef window, int x, int y, double scale=1.0) const;
        void        blend(WindowRef window, 
                          const Rect* device_rect=0, 
                          const Rect* bmp_rect=0) const;
        void        blend(WindowRef window, int x, int y, double scale=1.0) const;

        unsigned char* buf();
        unsigned       width() const;
        unsigned       height() const;
        int            row_bytes() const;
        unsigned       bpp() const { return m_bpp; }

        //Auxiliary static functions
        static unsigned calc_row_len(unsigned width, unsigned bits_per_pixel);
    private:
        pixel_map(const pixel_map&);
        const pixel_map& operator = (const pixel_map&);

    private:
        GWorldPtr      m_pmap;
        unsigned char* m_buf;
        unsigned       m_bpp;
        unsigned       m_img_size;
    };

}


#endif
