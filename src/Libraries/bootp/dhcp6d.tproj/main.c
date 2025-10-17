/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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
 * main.c
 * - stateless DHCPv6 server main
 */
/* 
 * Modification History
 *
 * September 7, 2018		Dieter Siegmund (dieter@apple.com)
 * - created
 */

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <dispatch/dispatch.h>
#include "DHCPv6Server.h"

int
main(int argc, char * argv[])
{
    const char *	config_file = NULL;
    DHCPv6ServerRef	server;
    dispatch_block_t	signal_block;
    dispatch_source_t	signal_source;

    if (argc > 1) {
	config_file = argv[1];
    }
    server = DHCPv6ServerCreate(config_file);
    if (server == NULL) {
	exit(1);
    }
    signal_source = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL,
					   SIGHUP,
					   0,
					   dispatch_get_main_queue());
    signal_block = ^{
	DHCPv6ServerUpdateConfiguration(server);
    };
    dispatch_source_set_event_handler(signal_source, signal_block);
    dispatch_resume(signal_source);
    signal(SIGHUP, SIG_IGN);
    dispatch_main();
    return (0);
}

