/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
extern const struct tok nlpid_values[];

#define	NLPID_NULLNS	  0x00
#define NLPID_Q933      0x08 /* ANSI T1.617 Annex D or ITU-T Q.933 Annex A */
#define NLPID_LMI       0x09 /* The original, aka Cisco, aka Gang of Four */
#define NLPID_SNAP      0x80
#define	NLPID_CLNP	    0x81 /* iso9577 */
#define	NLPID_ESIS	    0x82 /* iso9577 */
#define	NLPID_ISIS	    0x83 /* iso9577 */
#define NLPID_CONS      0x84
#define NLPID_IDRP      0x85
#define NLPID_MFR       0xb1 /* FRF.15 */
#define NLPID_SPB       0xc1 /* IEEE 802.1aq/D4.5 */
#define NLPID_IP        0xcc
#define NLPID_PPP       0xcf
#define NLPID_X25_ESIS  0x8a
#define NLPID_IP6       0x8e
