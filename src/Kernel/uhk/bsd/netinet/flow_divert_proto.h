/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#ifndef __FLOW_DIVERT_PROTO_H__
#define __FLOW_DIVERT_PROTO_H__

#define FLOW_DIVERT_CONTROL_NAME                "com.apple.flow-divert"

#define FLOW_DIVERT_TLV_LENGTH_UINT32   1

#define FLOW_DIVERT_PKT_CONNECT                 1
#define FLOW_DIVERT_PKT_CONNECT_RESULT          2
#define FLOW_DIVERT_PKT_DATA                    3
#define FLOW_DIVERT_PKT_CLOSE                   4
#define FLOW_DIVERT_PKT_READ_NOTIFY             5
#define FLOW_DIVERT_PKT_GROUP_INIT              6
#define FLOW_DIVERT_PKT_PROPERTIES_UPDATE       7
#define FLOW_DIVERT_PKT_APP_MAP_CREATE          9
#define FLOW_DIVERT_PKT_FLOW_STATES_REQUEST     10
#define FLOW_DIVERT_PKT_FLOW_STATES             11

#define FLOW_DIVERT_TLV_NIL                     0
#define FLOW_DIVERT_TLV_ERROR_CODE              5
#define FLOW_DIVERT_TLV_HOW                     7
#define FLOW_DIVERT_TLV_READ_COUNT              8
#define FLOW_DIVERT_TLV_SPACE_AVAILABLE         9
#define FLOW_DIVERT_TLV_CTL_UNIT                10
#define FLOW_DIVERT_TLV_LOCAL_ADDR              11
#define FLOW_DIVERT_TLV_REMOTE_ADDR             12
#define FLOW_DIVERT_TLV_OUT_IF_INDEX            13
#define FLOW_DIVERT_TLV_TRAFFIC_CLASS           14
#define FLOW_DIVERT_TLV_NO_CELLULAR             15
#define FLOW_DIVERT_TLV_FLOW_ID                 16
#define FLOW_DIVERT_TLV_TOKEN_KEY               17
#define FLOW_DIVERT_TLV_HMAC                    18
#define FLOW_DIVERT_TLV_KEY_UNIT                19
#define FLOW_DIVERT_TLV_LOG_LEVEL               20
#define FLOW_DIVERT_TLV_TARGET_HOSTNAME         21
#define FLOW_DIVERT_TLV_TARGET_ADDRESS          22
#define FLOW_DIVERT_TLV_TARGET_PORT             23
#define FLOW_DIVERT_TLV_CDHASH                  24
#define FLOW_DIVERT_TLV_SIGNING_ID              25
#define FLOW_DIVERT_TLV_AGGREGATE_UNIT          26
#define FLOW_DIVERT_TLV_IS_FRAGMENT             27
#define FLOW_DIVERT_TLV_PREFIX_COUNT            28
#define FLOW_DIVERT_TLV_FLAGS                   29
#define FLOW_DIVERT_TLV_FLOW_TYPE               30
#define FLOW_DIVERT_TLV_APP_DATA                31
#define FLOW_DIVERT_TLV_APP_AUDIT_TOKEN         32
#define FLOW_DIVERT_TLV_APP_REAL_SIGNING_ID     33
#define FLOW_DIVERT_TLV_APP_REAL_CDHASH         34
#define FLOW_DIVERT_TLV_APP_REAL_AUDIT_TOKEN    35
#define FLOW_DIVERT_TLV_CFIL_ID                 36
#define FLOW_DIVERT_TLV_DATAGRAM_SIZE           37
#define FLOW_DIVERT_TLV_ORDER                   38
#define FLOW_DIVERT_TLV_FLOW_STATE              39

#define FLOW_DIVERT_FLOW_TYPE_TCP               1
#define FLOW_DIVERT_FLOW_TYPE_UDP               3

#define FLOW_DIVERT_CHUNK_SIZE                  65600

#define FLOW_DIVERT_TOKEN_GETOPT_MAX_SIZE       128

#define FLOW_DIVERT_TOKEN_FLAG_VALIDATED        0x0000001
#define FLOW_DIVERT_TOKEN_FLAG_TFO              0x0000002
#define FLOW_DIVERT_TOKEN_FLAG_MPTCP            0x0000004
#define FLOW_DIVERT_TOKEN_FLAG_BOUND            0x0000008

#define FLOW_DIVERT_GROUP_FLAG_NO_APP_MAP       0x0000001
#define FLOW_DIVERT_GROUP_FLAG_DEFUNCT          0x0000002

#define FLOW_DIVERT_IS_TRANSPARENT              0x80000000

// Used for policies as well as opening control sockets
#define FLOW_DIVERT_IN_PROCESS_UNIT             0x0FFFFFFF

// Range for actual assigned control units
#define FLOW_DIVERT_IN_PROCESS_UNIT_MIN         0x0000FFFF
#define FLOW_DIVERT_IN_PROCESS_UNIT_MAX         0xFFFFFFFF

struct flow_divert_packet_header {
	uint8_t             packet_type;
	uint32_t            conn_id;
};

struct flow_divert_flow_state {
	uint32_t conn_id;
	uint64_t bytes_written_by_app;
	uint64_t bytes_sent;
	uint64_t bytes_received;
	uint32_t send_window;
	uint32_t send_buffer_bytes;
};

#endif /* __FLOW_DIVERT_PROTO_H__ */
