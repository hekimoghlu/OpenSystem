/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#include <dce/rpc.h>
/* For dce_get_802_addr() [on some platforms] */
#ifndef IEEE_802_FILE
#define IEEE_802_FILE   "/etc/ieee_802_addr"
#endif

#define utils_s_802_cant_read 0x1460101e
#define utils_s_802_addr_format 0x1460101f

typedef struct dce_802_addr_s_t {
    unsigned_char_t	eaddr[6];
} dce_802_addr_t;

void dce_get_802_addr(dce_802_addr_t*, error_status_t*);
