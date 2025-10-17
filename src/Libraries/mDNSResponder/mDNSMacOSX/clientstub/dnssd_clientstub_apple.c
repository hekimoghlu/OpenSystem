/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#include "dnssd_clientstub_apple.h"

#include "dns_sd_private.h"
#include "dnssd_clientstub.h"

#include <CoreUtils/CommonServices.h>
#include <CoreUtils/DebugServices.h>
#include <os/lock.h>
#include "mdns_strict.h"

#define kDNSServiceDefaultsKey_RequirePrivacy          "require_privacy"
#define kDNSServiceDefaultsKey_ResolverConfigPListData "resolver_config_plist_data"

static xpc_object_t		g_defaults_dict = NULL;
static os_unfair_lock	g_defaults_lock = OS_UNFAIR_LOCK_INIT;

DNSServiceErrorType DNSSD_API
DNSServiceSetResolverDefaults(const void * const plist_data_ptr, const size_t plist_data_len, bool require_privacy)
{
	DNSServiceErrorType err;
	require_action_quiet(plist_data_len <= (32 * CUBytesPerKibiByte), exit, err = kDNSServiceErr_Invalid);

	const xpc_object_t new_dict = xpc_dictionary_create(NULL, NULL, 0);
	require_action_quiet(new_dict, exit, err = kDNSServiceErr_NoMemory);

	xpc_dictionary_set_bool(new_dict, kDNSServiceDefaultsKey_RequirePrivacy, require_privacy);
	if (plist_data_ptr && (plist_data_len > 0)) {
		xpc_dictionary_set_data(new_dict, kDNSServiceDefaultsKey_ResolverConfigPListData, plist_data_ptr,
			plist_data_len);
	}
	os_unfair_lock_lock(&g_defaults_lock);
	xpc_object_t old_dict = g_defaults_dict;
	g_defaults_dict = new_dict;
	os_unfair_lock_unlock(&g_defaults_lock);

	xpc_forget(&old_dict);
	err = kDNSServiceErr_NoError;

exit:
	return err;
}

DNSServiceErrorType DNSSD_API
DNSServiceBrowseEx(DNSServiceRef *sdRef, DNSServiceFlags flags,
	uint32_t interfaceIndex, const char *regtype, const char *domain, const DNSServiceAttribute * const attr,
	DNSServiceBrowseReply callBack, void *context)
{
	return DNSServiceBrowseInternal(sdRef, flags, interfaceIndex, regtype, domain, attr, callBack, context);
}

DNSServiceErrorType DNSSD_API
DNSServiceResolveEx(DNSServiceRef *sdRef, DNSServiceFlags flags,
	uint32_t interfaceIndex, const char *name, const char *regtype, const char *domain,
	const DNSServiceAttribute * const attr, DNSServiceResolveReply callBack, void *context)
{
	return DNSServiceResolveInternal(sdRef, flags, interfaceIndex, name, regtype, domain, attr, callBack, context);
}

DNSServiceErrorType DNSSD_API
DNSServiceGetAddrInfoEx(DNSServiceRef *sdRef, DNSServiceFlags flags,
	uint32_t interfaceIndex, DNSServiceProtocol protocol, const char *hostname, const DNSServiceAttribute * const attr,
	DNSServiceGetAddrInfoReply callBack, void *context)
{
	return DNSServiceGetAddrInfoInternal(sdRef, flags, interfaceIndex, protocol, hostname, attr, callBack, context);
}

DNSServiceErrorType DNSSD_API
DNSServiceQueryRecordEx(DNSServiceRef * const sdRef, const DNSServiceFlags flags, const uint32_t ifindex,
	const char * const name, const uint16_t rrtype, const uint16_t rrclass, const DNSServiceAttribute * const attr,
	const DNSServiceQueryRecordReply callback, void * const context)
{
	return DNSServiceQueryRecordInternal(sdRef, flags, ifindex, name, rrtype, rrclass, attr, callback, context);
}

xpc_object_t
DNSServiceGetRetainedResolverDefaults(void)
{
	os_unfair_lock_lock(&g_defaults_lock);
	const xpc_object_t dict_copy = g_defaults_dict ? xpc_retain(g_defaults_dict) : NULL;
	os_unfair_lock_unlock(&g_defaults_lock);
	return dict_copy;
}

DNSServiceAttrRef
DNSServiceAttrCreate(void)
{
	return DNSServiceAttributeCreate();
}

void
DNSServiceAttrFree(DNSServiceAttrRef attr)
{
	DNSServiceAttributeDeallocate(attr);
}

size_t
get_required_tlv_length_for_defaults(const xpc_object_t defaults)
{
	size_t len = 0;
	size_t plist_len = 0;
	// Add length for IPC_TLV_TYPE_RESOLVER_CONFIG_PLIST_DATA.
	if (xpc_dictionary_get_data(defaults, kDNSServiceDefaultsKey_ResolverConfigPListData, &plist_len)) {
		len += get_required_tlv_length((uint16_t)plist_len);
	}
	// Add length for IPC_TLV_TYPE_REQUIRE_PRIVACY.
	len += get_required_tlv_uint8_length();
	return len;
}

const uint8_t *
get_validation_data_from_tlvs(const uint8_t * const ptr, const uint8_t * const limit, size_t * const length)
{
	size_t len = 0;
	const uint8_t *const data = get_tlv(ptr, limit, IPC_TLV_TYPE_SERVICE_ATTR_VALIDATION_DATA, &len);
	if (length)
	{
		*length = len;
	}
	return data;
}

const char *
get_tracker_hostname_from_tlvs(const uint8_t * const ptr, const uint8_t * const limit)
{
	return get_tlv_string(ptr, limit, IPC_TLV_TYPE_SERVICE_ATTR_TRACKER_STR);
}

void
put_tlvs_for_defaults(const xpc_object_t defaults, ipc_msg_hdr * const hdr, uint8_t ** const ptr,
	const uint8_t * const limit)
{
	size_t len = 0;
	const uint8_t * const data = xpc_dictionary_get_data(defaults, kDNSServiceDefaultsKey_ResolverConfigPListData,
		&len);
	if (data) {
		put_tlv(IPC_TLV_TYPE_RESOLVER_CONFIG_PLIST_DATA, (uint16_t)len, data, ptr, limit);
	}
	const uint8_t require_privacy = xpc_dictionary_get_bool(defaults, kDNSServiceDefaultsKey_RequirePrivacy) ? 1 : 0;
	put_tlv_uint8(IPC_TLV_TYPE_REQUIRE_PRIVACY, require_privacy, ptr, limit);
	hdr->ipc_flags |= IPC_FLAGS_TRAILING_TLVS;
}
