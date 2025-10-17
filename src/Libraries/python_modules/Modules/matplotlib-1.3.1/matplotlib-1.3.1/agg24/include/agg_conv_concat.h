/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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

#ifndef AGG_CONV_CONCAT_INCLUDED
#define AGG_CONV_CONCAT_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //=============================================================conv_concat
    // Concatenation of two paths. Usually used to combine lines or curves 
    // with markers such as arrowheads
    template<class VS1, class VS2> class conv_concat
    {
    public:
        conv_concat(VS1& source1, VS2& source2) :
            m_source1(&source1), m_source2(&source2), m_status(2) {}
        void attach1(VS1& source) { m_source1 = &source; }
        void attach2(VS2& source) { m_source2 = &source; }


        void rewind(unsigned path_id)
        { 
            m_source1->rewind(path_id);
            m_source2->rewind(0);
            m_status = 0;
        }

        unsigned vertex(double* x, double* y)
        {
            unsigned cmd;
            if(m_status == 0)
            {
                cmd = m_source1->vertex(x, y);
                if(!is_stop(cmd)) return cmd;
                m_status = 1;
            }
            if(m_status == 1)
            {
                cmd = m_source2->vertex(x, y);
                if(!is_stop(cmd)) return cmd;
                m_status = 2;
            }
            return path_cmd_stop;
        }

    private:
        conv_concat(const conv_concat<VS1, VS2>&);
        const conv_concat<VS1, VS2>& 
            operator = (const conv_concat<VS1, VS2>&);

        VS1* m_source1;
        VS2* m_source2;
        int  m_status;

    };
}


#endif
