/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#ifndef network_cmds_lib_h
#define network_cmds_lib_h

/*
 * @function   clean_non_printable
 * @discussion Modifies a string to replace the non-printable ASCII characters
 *             with '?'
 * @param str  The string to be cleaned up
 * @param len  The length of the string
 * @result     Returns 'str'
 */
extern char *clean_non_printable(char *str, size_t len);

/*
 * @function   dump_hex
 * @discussion Dump hex bytes to stdout
 * @param ptr  The buffer to dump
 * @param len  The length of the string
 */
extern void dump_hex(const unsigned char *ptr, size_t len);


extern uint16_t in_cksum(uint16_t *addr, uint16_t len);

#endif /* network_cmds_lib_h */
