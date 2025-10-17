/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
 * DHCPv6Server.h
 * - stateless DHCPv6 server
 */
/* 
 * Modification History
 *
 * August 28, 2018		Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _S_DHCPV6SERVER_H
#define _S_DHCPV6SERVER_H

#include <stdint.h>
#include "DHCPv6.h"

typedef struct DHCPv6Server * DHCPv6ServerRef;

void
DHCPv6ServerSetVerbose(bool verbose);

void
DHCPv6ServerSetPorts(uint16_t client_port, uint16_t server_port);

DHCPv6ServerRef
DHCPv6ServerCreate(const char * config_file);

void
DHCPv6ServerUpdateConfiguration(DHCPv6ServerRef server);

#endif /* _S_DHCPV6SERVER_H */
