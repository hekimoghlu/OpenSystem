/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#ifndef _EAP8021X_CHAP_H
#define _EAP8021X_CHAP_H

#include <stdint.h>

/* 
 * Modification History
 *
 * December 10, 2001	Dieter Siegmund (dieter@apple.com)
 * - created
 */

/*
 * Function: chap_md5
 * Purpose:
 *   Compute the CHAP MD5 hash using the method described in
 *   RFC 1994.
 */
void
chap_md5(uint8_t identifier, const uint8_t * password, int password_length,
	 const uint8_t * challenge, int challenge_len,
	 uint8_t * hash);
#endif /* _EAP802_1X_CHAP_H */
