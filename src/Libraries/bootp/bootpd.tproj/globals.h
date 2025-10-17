/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#ifndef _S_GLOBALS_H
#define _S_GLOBALS_H

#include "subnets.h"

extern int		bootp_socket;
extern bool		debug;
extern bool		dhcp_ignore_client_identifier;
extern bool		dhcp_supply_bootfile;
extern int		quiet;
extern unsigned short	server_priority;
extern uint32_t		reply_threshold_seconds;
extern const uint8_t	rfc_magic[4];
extern char		server_name[MAXHOSTNAMELEN + 1];
extern SubnetListRef	subnets;
extern char *		testing_control;
extern char *		transmit_buffer;
extern bool		use_open_directory;
extern bool		verbose;
#endif /* _S_GLOBALS_H */
