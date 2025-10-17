/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#import <Network/Network.h>

#if USE(APPLE_INTERNAL_SDK)

#import <nw/private.h>

#else

WTF_EXTERN_C_BEGIN

void nw_parameters_set_account_id(nw_parameters_t, const char * account_id);
void nw_parameters_set_source_application(nw_parameters_t, audit_token_t);
void nw_parameters_set_source_application_by_bundle_id(nw_parameters_t, const char*);
void nw_parameters_set_attributed_bundle_identifier(nw_parameters_t, const char*);
nw_endpoint_t nw_endpoint_create_host_with_numeric_port(const char* hostname, uint16_t port_host_order);
const char* nw_endpoint_get_known_tracker_name(nw_endpoint_t);
bool nw_nat64_does_interface_index_support_nat64(uint32_t ifindex);

void nw_parameters_set_is_third_party_web_content(nw_parameters_t, bool is_third_party_web_content);
void nw_parameters_set_is_known_tracker(nw_parameters_t, bool is_known_tracker);
void nw_parameters_allow_sharing_port_with_listener(nw_parameters_t, nw_listener_t);

#define SO_TC_BK_SYS 100
#define SO_TC_BE 0
#define SO_TC_VI 700
#define SO_TC_VO 800

void nw_connection_reset_traffic_class(nw_connection_t, uint32_t traffic_class);
void nw_parameters_set_traffic_class(nw_parameters_t, uint32_t traffic_class);

nw_interface_t nw_interface_create_with_name(const char *interface_name);
nw_interface_t nw_path_copy_interface(nw_path_t);

bool nw_settings_get_unified_http_enabled(void);

void nw_parameters_set_server_mode(nw_parameters_t, bool);
nw_parameters_t nw_parameters_create_webtransport_http(nw_parameters_configure_protocol_block_t, nw_parameters_configure_protocol_block_t, nw_parameters_configure_protocol_block_t, nw_parameters_configure_protocol_block_t);
nw_protocol_options_t nw_webtransport_create_options(void);
bool nw_protocol_options_is_webtransport(nw_protocol_options_t);
void nw_webtransport_options_set_is_unidirectional(nw_protocol_options_t, bool);
void nw_webtransport_options_set_is_datagram(nw_protocol_options_t, bool);
void nw_webtransport_options_set_connection_max_sessions(nw_protocol_options_t, uint64_t);
void nw_webtransport_options_add_connect_request_header(nw_protocol_options_t, const char*, const char*);
uint32_t nw_webtransport_metadata_get_session_error_code(nw_protocol_metadata_t);
void nw_webtransport_metadata_set_session_error_code(nw_protocol_metadata_t, uint32_t);
const char* nw_webtransport_metadata_get_session_error_message(nw_protocol_metadata_t);
void nw_webtransport_metadata_set_session_error_message(nw_protocol_metadata_t, const char*);

WTF_EXTERN_C_END

#endif // USE(APPLE_INTERNAL_SDK)
