/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#ifndef status_exit_codes_h
#define status_exit_codes_h

/* S_ERR_ND_* are libnetdissect status */

typedef enum {
	S_SUCCESS           = 0, /* not a libnetdissect status */
	S_ERR_HOST_PROGRAM  = 1, /* not a libnetdissect status */
	S_ERR_ND_NO_PRINTER = 11,
	S_ERR_ND_MEM_ALLOC  = 12,
	S_ERR_ND_OPEN_FILE  = 13,
	S_ERR_ND_WRITE_FILE = 14,
	S_ERR_ND_ESP_SECRET = 15
} status_exit_codes_t;

#endif /* status_exit_codes_h */
