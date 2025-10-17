/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#ifndef _KDP_SERIAL_H_
#define _KDP_SERIAL_H_

/*
 * APIs for escaping a KDP UDP packet into a byte stream suitable
 * for a standard serial console
 */

enum {SERIALIZE_WAIT_START, SERIALIZE_READING};

/*
 * Take a buffer of specified length and output it with the given
 * function. Escapes special characters as needed
 */
void kdp_serialize_packet(unsigned char *, unsigned int, void (*func)(char));

/*
 * Add a new character to an internal buffer, and return that
 * buffer when a fully constructed packet has been identified.
 * Will track intermediate state using magic enums above
 */
unsigned char *kdp_unserialize_packet(unsigned char, unsigned int *);

#endif
