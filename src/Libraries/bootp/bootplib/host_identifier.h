/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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

#ifndef _S_HOST_IDENTIFIER_H
#define _S_HOST_IDENTIFIER_H
/*
 * Copyright (c) 1999 Apple Inc. All rights reserved.
 *
 * @APPLE_LICENSE_HEADER_START@
 * 
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 * 
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 * 
 * @APPLE_LICENSE_HEADER_END@
 */
/*
 * host_identifier.h
 * - the identifier string format is:
 *   		%x,%x:%x...:%x
 * - the byte before the slash '/' is the arp hardware type (or 0),
 *   the bytes following the slash are the hardware address bytes, or
 *   the opaque client identifier bytes
 * - each returns NULL if a failure occurred, a non-NULL pointer otherwise;
 *   the non-NULL pointer may be freed using free()
 * - arp hardware types are defined in the "Assigned Numbers RFC",
 *   currently this is RFC 1700 (look for "hrd")
 *   
 */

/*
 * Modification History
 *
 * June 4, 1998 Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

#include <stdint.h>

char *
identifierToString(uint8_t type, const void * identifier, int len);

char *
identifierToStringWithBuffer(uint8_t type, const void * identifier, int len,
			     char * buf, int buf_len);

void *
identifierFromString(const char * str, uint8_t * type, int * len);

#endif /* _S_HOST_IDENTIFIER_H */
