/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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
#ifndef MODULES_AUDIO_CODING_CODECS_PCM16B_PCM16B_H_
#define MODULES_AUDIO_CODING_CODECS_PCM16B_PCM16B_H_
/*
 * Define the fixpoint numeric formats
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************
 * WebRtcPcm16b_Encode(...)
 *
 * "Encode" a sample vector to 16 bit linear (Encoded standard is big endian)
 *
 * Input:
 *              - speech        : Input speech vector
 *              - len           : Number of samples in speech vector
 *
 * Output:
 *              - encoded       : Encoded data vector (big endian 16 bit)
 *
 * Returned value               : Length (in bytes) of coded data.
 *                                Always equal to twice the len input parameter.
 */

size_t WebRtcPcm16b_Encode(const int16_t* speech, size_t len, uint8_t* encoded);

/****************************************************************************
 * WebRtcPcm16b_Decode(...)
 *
 * "Decode" a vector to 16 bit linear (Encoded standard is big endian)
 *
 * Input:
 *              - encoded       : Encoded data vector (big endian 16 bit)
 *              - len           : Number of bytes in encoded
 *
 * Output:
 *              - speech        : Decoded speech vector
 *
 * Returned value               : Samples in speech
 */

size_t WebRtcPcm16b_Decode(const uint8_t* encoded, size_t len, int16_t* speech);

#ifdef __cplusplus
}
#endif

#endif /* MODULES_AUDIO_CODING_CODECS_PCM16B_PCM16B_H_ */
