/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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
// Filtering class image_filter_lut implemantation
//
//----------------------------------------------------------------------------


#include "agg_image_filters.h"


namespace agg
{
    //--------------------------------------------------------------------
    void image_filter_lut::realloc_lut(double radius)
    {
        m_radius = radius;
        m_diameter = uceil(radius) * 2;
        m_start = -int(m_diameter / 2 - 1);
        unsigned size = m_diameter << image_subpixel_shift;
        if(size > m_weight_array.size())
        {
            m_weight_array.resize(size);
        }
    }



    //--------------------------------------------------------------------
    // This function normalizes integer values and corrects the rounding 
    // errors. It doesn't do anything with the source floating point values
    // (m_weight_array_dbl), it corrects only integers according to the rule 
    // of 1.0 which means that any sum of pixel weights must be equal to 1.0.
    // So, the filter function must produce a graph of the proper shape.
    //--------------------------------------------------------------------
    void image_filter_lut::normalize()
    {
        unsigned i;
        int flip = 1;

        for(i = 0; i < image_subpixel_scale; i++)
        {
            for(;;)
            {
                int sum = 0;
                unsigned j;
                for(j = 0; j < m_diameter; j++)
                {
                    sum += m_weight_array[j * image_subpixel_scale + i];
                }

                if(sum == image_filter_scale) break;

                double k = double(image_filter_scale) / double(sum);
                sum = 0;
                for(j = 0; j < m_diameter; j++)
                {
                    sum +=     m_weight_array[j * image_subpixel_scale + i] = 
                        iround(m_weight_array[j * image_subpixel_scale + i] * k);
                }

                sum -= image_filter_scale;
                int inc = (sum > 0) ? -1 : 1;

                for(j = 0; j < m_diameter && sum; j++)
                {
                    flip ^= 1;
                    unsigned idx = flip ? m_diameter/2 + j/2 : m_diameter/2 - j/2;
                    int v = m_weight_array[idx * image_subpixel_scale + i];
                    if(v < image_filter_scale)
                    {
                        m_weight_array[idx * image_subpixel_scale + i] += inc;
                        sum += inc;
                    }
                }
            }
        }

        unsigned pivot = m_diameter << (image_subpixel_shift - 1);

        for(i = 0; i < pivot; i++)
        {
            m_weight_array[pivot + i] = m_weight_array[pivot - i];
        }
        unsigned end = (diameter() << image_subpixel_shift) - 1;
        m_weight_array[0] = m_weight_array[end];
    }


}

