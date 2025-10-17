/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#include "dns_sd_private.h"
#include "dnssd_ipc.h"

#include <xpc/xpc.h>

xpc_object_t
DNSServiceGetRetainedResolverDefaults(void);

size_t
get_required_tlv_length_for_validation_attr(const DNSServiceAttribute *attr);

size_t
get_required_tlv_length_for_defaults(xpc_object_t defaults);

size_t
get_required_tlv_length_for_get_tracker_info(void);

const uint8_t *
get_validation_data_from_tlvs(const uint8_t * const ptr, const uint8_t * const limit, size_t * const length);

const char *
get_tracker_hostname_from_tlvs(const uint8_t * const ptr, const uint8_t * const limit);

void
put_tlvs_for_validation_attr(const DNSServiceAttribute * const attr, ipc_msg_hdr * const hdr, uint8_t ** const ptr,
	const uint8_t * const limit);

void
put_tlvs_for_defaults(xpc_object_t defaults, ipc_msg_hdr *hdr, uint8_t **ptr, const uint8_t *limit);

void
put_tlv_to_get_tracker_info(ipc_msg_hdr * const hdr, uint8_t ** const ptr, const uint8_t * const limit);
