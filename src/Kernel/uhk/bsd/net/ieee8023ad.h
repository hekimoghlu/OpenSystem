/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
/*
 * ieee8023ad.h
 */

/*
 * Modification History
 *
 * May 14, 2004	Dieter Siegmund (dieter@apple.com)
 * - created
 */


#ifndef _NET_IEEE8023AD_H_
#define _NET_IEEE8023AD_H_

#include <sys/types.h>

#define IEEE8023AD_SLOW_PROTO_ETHERTYPE                         0x8809
#define IEEE8023AD_SLOW_PROTO_MULTICAST { 0x01, 0x80, 0xc2, 0x00, 0x00, 0x02 }

#define IEEE8023AD_SLOW_PROTO_SUBTYPE_LACP                      1
#define IEEE8023AD_SLOW_PROTO_SUBTYPE_LA_MARKER_PROTOCOL        2
#define IEEE8023AD_SLOW_PROTO_SUBTYPE_RESERVED_START            3
#define IEEE8023AD_SLOW_PROTO_SUBTYPE_RESERVED_END              10
#endif /* _NET_IEEE8023AD_H_ */
