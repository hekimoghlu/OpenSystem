/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
/* Types missing from some systems */

/*
 * Network layer prototocol identifiers
 */
#ifndef ISO8473_CLNP
#define ISO8473_CLNP		0x81
#endif
#ifndef	ISO9542_ESIS
#define	ISO9542_ESIS		0x82
#endif
#ifndef ISO9542X25_ESIS
#define ISO9542X25_ESIS		0x8a
#endif
#ifndef	ISO10589_ISIS
#define	ISO10589_ISIS		0x83
#endif
/*
 * this does not really belong in the nlpid.h file
 * however we need it for generating nice
 * IS-IS related BPF filters
 */
#define ISIS_L1_LAN_IIH      15
#define ISIS_L2_LAN_IIH      16
#define ISIS_PTP_IIH         17
#define ISIS_L1_LSP          18
#define ISIS_L2_LSP          20
#define ISIS_L1_CSNP         24
#define ISIS_L2_CSNP         25
#define ISIS_L1_PSNP         26
#define ISIS_L2_PSNP         27

#ifndef ISO8878A_CONS
#define	ISO8878A_CONS		0x84
#endif
#ifndef	ISO10747_IDRP
#define	ISO10747_IDRP		0x85
#endif
