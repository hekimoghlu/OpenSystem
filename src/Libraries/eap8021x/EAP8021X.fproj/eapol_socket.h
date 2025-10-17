/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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
 * eapol_socket.h
 * - wrapper for allocating an NDRV socket for use with 802.1X
 */

/* 
 * Modification History
 *
 * October 26, 2001	Dieter Siegmund (dieter@apple)
 * - created
 *
 * August 31, 2010	Dieter Siegmund (dieter@apple.com)
 * - combined ndrv_socket.h/eapol_socket.h, moved to framework
 */


#ifndef _S_EAPOL_SOCKET_H
#define _S_EAPOL_SOCKET_H

#include <stdbool.h>

int
eapol_socket(const char * ifname, bool is_wireless);

#endif /* _S_EAPOL_SOCKET_H */

