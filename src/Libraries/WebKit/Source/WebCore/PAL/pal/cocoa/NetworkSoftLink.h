/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
#pragma once

#if HAVE(WEB_TRANSPORT)

#import <pal/spi/cocoa/NetworkSPI.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, Network)

// FIXME: remove this soft linking once rdar://141009498 is available on all tested OS builds.
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, Network, nw_webtransport_options_add_connect_request_header, void, (nw_protocol_options_t options, const char* name, const char* value), (options, name, value))
#define nw_webtransport_options_add_connect_request_header PAL::softLink_Network_nw_webtransport_options_add_connect_request_header

// FIXME: remove this soft linking once rdar://141715299 is available on all tested OS builds.
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, Network, nw_webtransport_metadata_get_session_error_code, uint32_t, (nw_protocol_metadata_t metadata), (metadata))
#define nw_webtransport_metadata_get_session_error_code PAL::softLink_Network_nw_webtransport_metadata_get_session_error_code

// FIXME: remove this soft linking once rdar://141715299 is available on all tested OS builds.
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, Network, nw_webtransport_metadata_set_session_error_code, void, (nw_protocol_metadata_t metadata, uint32_t session_error_code), (metadata, session_error_code))
#define nw_webtransport_metadata_set_session_error_code PAL::softLink_Network_nw_webtransport_metadata_set_session_error_code

// FIXME: remove this soft linking once rdar://141715299 is available on all tested OS builds.
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, Network, nw_webtransport_metadata_get_session_error_message, const char*, (nw_protocol_metadata_t metadata), (metadata))
#define nw_webtransport_metadata_get_session_error_message PAL::softLink_Network_nw_webtransport_metadata_get_session_error_message

// FIXME: remove this soft linking once rdar://141715299 is available on all tested OS builds.
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, Network, nw_webtransport_metadata_set_session_error_message, void, (nw_protocol_metadata_t metadata, const char* session_error_message), (metadata, session_error_message))
#define nw_webtransport_metadata_set_session_error_message PAL::softLink_Network_nw_webtransport_metadata_set_session_error_message

#endif // HAVE(WEB_TRANSPORT)
