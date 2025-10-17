/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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

//
//  test_utils.h
//  DCERPCTests
//
//  Created by William Conway on 12/1/23.
//

#ifndef test_utils_h
#define test_utils_h

#include <compat/dcerpc.h>

// Obtain an rpc binding handle to the
// given host, protocol, and endpoint.
//
// hostname: IP address, localhost, hostname
// protocol: ncacn_ip_tcp, ncacn_ip_udp.
// endpoint: tcp or udp port number.
//
// Returns:
//     1 on success
//     0 on failure
//
int get_client_rpc_binding(
    rpc_binding_handle_t * binding_handle,
    const char * hostname,
    const char * protocol,
    const char * endpoint);

// Prints a description of a given error_status_t code.
// Arguments
//     ecode:   an rpc error_status_t code.
//     routine: Otional routine name which returned the error.
//     ctx:    Optional context about the error.
//     fatal:   Calls exit(1) if not set to zero.
//
void chk_dce_err(error_status_t ecode, const char *routine, const char *ctx, unsigned int fatal);

// Returns true if each string is the reverse of the other.
// Returns false otherwise.
bool stringsAreReversed(const char *str1, const char *str2);

#endif /* test_utils_h */
