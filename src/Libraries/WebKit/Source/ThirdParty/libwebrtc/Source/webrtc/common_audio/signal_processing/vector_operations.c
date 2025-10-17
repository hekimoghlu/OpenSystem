/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#include "common_audio/signal_processing/include/signal_processing_library.h"

void WebRtcSpl_ReverseOrderMultArrayElements(int16_t *out, const int16_t *in,
                                             const int16_t *win,
                                             size_t vector_length,
                                             int16_t right_shifts)
{
    size_t i;
    int16_t *outptr = out;
    const int16_t *inptr = in;
    const int16_t *winptr = win;
    for (i = 0; i < vector_length; i++)
    {
      *outptr++ = (int16_t)((*inptr++ * *winptr--) >> right_shifts);
    }
}

void WebRtcSpl_ElementwiseVectorMult(int16_t *out, const int16_t *in,
                                     const int16_t *win, size_t vector_length,
                                     int16_t right_shifts)
{
    size_t i;
    int16_t *outptr = out;
    const int16_t *inptr = in;
    const int16_t *winptr = win;
    for (i = 0; i < vector_length; i++)
    {
      *outptr++ = (int16_t)((*inptr++ * *winptr++) >> right_shifts);
    }
}

void WebRtcSpl_AddVectorsAndShift(int16_t *out, const int16_t *in1,
                                  const int16_t *in2, size_t vector_length,
                                  int16_t right_shifts)
{
    size_t i;
    int16_t *outptr = out;
    const int16_t *in1ptr = in1;
    const int16_t *in2ptr = in2;
    for (i = vector_length; i > 0; i--)
    {
        (*outptr++) = (int16_t)(((*in1ptr++) + (*in2ptr++)) >> right_shifts);
    }
}

void WebRtcSpl_AddAffineVectorToVector(int16_t *out, const int16_t *in,
                                       int16_t gain, int32_t add_constant,
                                       int16_t right_shifts,
                                       size_t vector_length)
{
    size_t i;

    for (i = 0; i < vector_length; i++)
    {
      out[i] += (int16_t)((in[i] * gain + add_constant) >> right_shifts);
    }
}

void WebRtcSpl_AffineTransformVector(int16_t *out, const int16_t *in,
                                     int16_t gain, int32_t add_constant,
                                     int16_t right_shifts, size_t vector_length)
{
    size_t i;

    for (i = 0; i < vector_length; i++)
    {
      out[i] = (int16_t)((in[i] * gain + add_constant) >> right_shifts);
    }
}
