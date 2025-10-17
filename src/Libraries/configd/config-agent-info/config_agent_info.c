/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#include "config_agent_info.h"
#include "configAgentDefines.h"
#include "network_config_agent_info_priv.h"

#include <syslog.h>

static void
get_agent_uuid_if_OOB_data_required(xpc_object_t info, uuid_t uuid)
{
	__block xpc_object_t agent_uuid = NULL;

	if (xpc_get_type(info) == XPC_TYPE_ARRAY) {
		xpc_array_apply(info, ^bool(size_t index, xpc_object_t value) {
#pragma unused(index)
			if ((value != NULL) &&
			    (xpc_get_type(value) == XPC_TYPE_DICTIONARY)) {
				agent_uuid = xpc_dictionary_get_value(value,
								      kConfigAgentOutOfBandDataUUID);
				if (agent_uuid != NULL) {
					return false;
				}
			}
			return true;
		});
	} else if (xpc_get_type(info) == XPC_TYPE_DICTIONARY) {
		agent_uuid = xpc_dictionary_get_value(info,
						      kConfigAgentOutOfBandDataUUID);
	}

	if ((agent_uuid != NULL) &&
	    (xpc_get_type(agent_uuid) == XPC_TYPE_DATA) &&
	    (xpc_data_get_length(agent_uuid) >= sizeof(uuid_t))) {
		const void *bytes = xpc_data_get_bytes_ptr(agent_uuid);
		uuid_copy(uuid, bytes);
	} else {
		uuid_clear(uuid);
	}
}

static boolean_t
is_a_config_agent(const struct netagent *agent)
{
	const char *agentDomain;

	if (agent == NULL) {
		return false;
	}

	agentDomain = agent->netagent_domain;
	if (agentDomain == NULL || strcmp(agentDomain, kConfigAgentDomain)) {
		return false;
	}

	return true;
}

boolean_t
is_config_agent_type_dns(const struct netagent *agent)
{
	const char *agentDesc;

	if (!is_a_config_agent(agent)) {
		return false;
	}

	agentDesc = agent->netagent_type;
	if (agentDesc == NULL || strcmp(agentDesc, kConfigAgentTypeDNS)) {
		return false;
	}

	return true;
}

boolean_t
is_config_agent_type_proxy(const struct netagent *agent)
{
	const char *agentDesc;

	if (!is_a_config_agent(agent)) {
		return false;
	}

	agentDesc = agent->netagent_type;
	if (agentDesc == NULL || strcmp(agentDesc, kConfigAgentTypeProxy)) {
		return false;
	}

	return true;
}

static boolean_t
is_config_agent_type_dns_multicast(const struct netagent *agent)
{
	if (strncmp(agent->netagent_desc, kConfigAgentTypeDNSMulticast, sizeof(kConfigAgentTypeDNSMulticast)-1) == 0) {
		return true;
	}

	return false;
}

static boolean_t
is_config_agent_type_dns_private(const struct netagent *agent)
{
	if (strncmp(agent->netagent_desc, kConfigAgentTypeDNSPrivate, sizeof(kConfigAgentTypeDNSPrivate)-1) == 0) {
		return true;
	}

	return false;
}

xpc_object_t
config_agent_copy_dns_information(const struct netagent *agent)
{
	xpc_object_t resolver = NULL;

	if (!is_config_agent_type_dns(agent)) {
		goto done;
	}

	if (agent->netagent_data_size <= 0 ) {
		if (!is_config_agent_type_dns_private(agent) && !is_config_agent_type_dns_multicast(agent)) {
			const char *agent_desc = (*(agent->netagent_desc) != '\0') ? agent->netagent_desc : kConfigAgentTypeDNS;
			syslog(LOG_ERR, "Cannot parse config agent (%s). No data available", agent_desc);
		}

		goto done;
	}

	resolver = xpc_create_from_plist(agent->netagent_data, agent->netagent_data_size);

done:
	return resolver;
}

xpc_object_t
config_agent_get_dns_nameservers(xpc_object_t resolver)
{
	if ((resolver == NULL) ||
	    (xpc_get_type(resolver) != XPC_TYPE_DICTIONARY)) {
		return NULL;
	}

	return xpc_dictionary_get_value(resolver, kConfigAgentDNSNameServers);
}

xpc_object_t
config_agent_get_dns_searchdomains(xpc_object_t resolver)
{
	if ((resolver == NULL) ||
	    (xpc_get_type(resolver) != XPC_TYPE_DICTIONARY)) {
		return NULL;
	}

	return xpc_dictionary_get_value(resolver, kConfigAgentDNSSearchDomains);
}

void
config_agent_free_dns_information(xpc_object_t resolver)
{
	if (resolver == NULL) {
		syslog(LOG_ERR, "Attempting to free invalid resolver");
		return;
	}

	xpc_release(resolver);
}

xpc_object_t
config_agent_copy_proxy_information(const struct netagent *agent)
{
	xpc_object_t info = NULL;

	if (!is_config_agent_type_proxy(agent)) {
		goto done;
	}

	if (agent->netagent_data_size <= 0 ) {
		const char *agent_desc = (*(agent->netagent_desc) != '\0') ? agent->netagent_desc : kConfigAgentTypeProxy;
		syslog(LOG_ERR, "Cannot parse config agent (%s). No data available", agent_desc);
		goto done;
	}

	info = xpc_create_from_plist(agent->netagent_data, agent->netagent_data_size);

done:
	return info;
}

xpc_object_t
config_agent_update_proxy_information(xpc_object_t proxyConfig)
{
	struct netagent	agent;
	xpc_object_t	newProxyConfig	= NULL;

	if (proxyConfig == NULL) {
		return NULL;
	}

	get_agent_uuid_if_OOB_data_required(proxyConfig, agent.netagent_uuid);

	if (uuid_is_null(agent.netagent_uuid) == 0) {
		void		*buffer;
		uint64_t	length;

		strlcpy(agent.netagent_type, kConfigAgentTypeProxy, sizeof(agent.netagent_type));
		buffer = _nwi_config_agent_copy_data(&agent, &length);
		if ((buffer != NULL) && (length > 0)) {
			newProxyConfig = xpc_create_from_plist(buffer, (size_t)length);
			free(buffer);
		}
	}

	return newProxyConfig;
}

void
config_agent_free_proxy_information(xpc_object_t proxyConfig)
{
	if (proxyConfig == NULL) {
		syslog(LOG_ERR, "Attempting to free proxy configuration");
		return;
	}

	xpc_release(proxyConfig);
}
