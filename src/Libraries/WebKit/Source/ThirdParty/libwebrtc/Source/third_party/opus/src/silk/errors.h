/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
#ifndef SILK_ERRORS_H
#define SILK_ERRORS_H

#ifdef __cplusplus
extern "C"
{
#endif

/******************/
/* Error messages */
/******************/
#define SILK_NO_ERROR                               0

/**************************/
/* Encoder error messages */
/**************************/

/* Input length is not a multiple of 10 ms, or length is longer than the packet length */
#define SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES        -101

/* Sampling frequency not 8000, 12000 or 16000 Hertz */
#define SILK_ENC_FS_NOT_SUPPORTED                   -102

/* Packet size not 10, 20, 40, or 60 ms */
#define SILK_ENC_PACKET_SIZE_NOT_SUPPORTED          -103

/* Allocated payload buffer too short */
#define SILK_ENC_PAYLOAD_BUF_TOO_SHORT              -104

/* Loss rate not between 0 and 100 percent */
#define SILK_ENC_INVALID_LOSS_RATE                  -105

/* Complexity setting not valid, use 0...10 */
#define SILK_ENC_INVALID_COMPLEXITY_SETTING         -106

/* Inband FEC setting not valid, use 0 or 1 */
#define SILK_ENC_INVALID_INBAND_FEC_SETTING         -107

/* DTX setting not valid, use 0 or 1 */
#define SILK_ENC_INVALID_DTX_SETTING                -108

/* CBR setting not valid, use 0 or 1 */
#define SILK_ENC_INVALID_CBR_SETTING                -109

/* Internal encoder error */
#define SILK_ENC_INTERNAL_ERROR                     -110

/* Internal encoder error */
#define SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR   -111

/**************************/
/* Decoder error messages */
/**************************/

/* Output sampling frequency lower than internal decoded sampling frequency */
#define SILK_DEC_INVALID_SAMPLING_FREQUENCY         -200

/* Payload size exceeded the maximum allowed 1024 bytes */
#define SILK_DEC_PAYLOAD_TOO_LARGE                  -201

/* Payload has bit errors */
#define SILK_DEC_PAYLOAD_ERROR                      -202

/* Payload has bit errors */
#define SILK_DEC_INVALID_FRAME_SIZE                 -203

#ifdef __cplusplus
}
#endif

#endif
