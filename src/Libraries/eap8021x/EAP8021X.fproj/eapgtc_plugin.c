/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
 * Modification History
 *
 * June 3, 2003	Dieter Siegmund (dieter@apple.com)
 * - created
 */
 
#include <EAP8021X/EAPClientPlugin.h>
#include <EAP8021X/EAPClientProperties.h>
#include <mach/boolean.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <syslog.h>

/*
 * Declare these here to ensure that the compiler
 * generates appropriate errors/warnings
 */
EAPClientPluginFuncIntrospect eapgtc_introspect;
static EAPClientPluginFuncVersion eapgtc_version;
static EAPClientPluginFuncEAPType eapgtc_type;
static EAPClientPluginFuncEAPName eapgtc_name;
static EAPClientPluginFuncInit eapgtc_init;
static EAPClientPluginFuncFree eapgtc_free;
static EAPClientPluginFuncRequireProperties eapgtc_require_props;
static EAPClientPluginFuncProcess eapgtc_process;
static EAPClientPluginFuncFreePacket eapgtc_free_packet;

#define EAPGTC_CHALLENGE_PREFIX 	"CHALLENGE="
#define EAPGTC_CHALLENGE_PREFIX_LENGTH	(sizeof(EAPGTC_CHALLENGE_PREFIX) - 1)
#define EAPGTC_RESPONSE_PREFIX 		"RESPONSE="
#define EAPGTC_RESPONSE_PREFIX_LENGTH	(sizeof(EAPGTC_RESPONSE_PREFIX) - 1)

static EAPPacketRef
eapgtc_request(EAPClientPluginDataRef plugin, 
	       const EAPRequestPacketRef in_pkt_p)
{
    int				in_length;
    EAPResponsePacketRef	out_pkt_p = NULL;
    bool			prefix = FALSE;
    int				size;

    size = sizeof(*out_pkt_p) + plugin->password_length;
    in_length = EAPPacketGetLength((EAPPacketRef)in_pkt_p) - sizeof(*in_pkt_p);
    if (in_length >= EAPGTC_CHALLENGE_PREFIX_LENGTH
	&& strncmp((const char *)in_pkt_p->type_data, EAPGTC_CHALLENGE_PREFIX,
		   EAPGTC_CHALLENGE_PREFIX_LENGTH) == 0) {
	prefix = TRUE;
	size += EAPGTC_RESPONSE_PREFIX_LENGTH + plugin->username_length + 1;
    }
    out_pkt_p = malloc(size);
    if (out_pkt_p == NULL) {
	goto failed;
    }
    out_pkt_p->code = kEAPCodeResponse;
    out_pkt_p->identifier = in_pkt_p->identifier;
    EAPPacketSetLength((EAPPacketRef)out_pkt_p, size);
    out_pkt_p->type = kEAPTypeGenericTokenCard;
    if (prefix) {
	int	offset = 0;

	memcpy(out_pkt_p->type_data, EAPGTC_RESPONSE_PREFIX,
	       EAPGTC_RESPONSE_PREFIX_LENGTH);
	offset += EAPGTC_RESPONSE_PREFIX_LENGTH;
	memcpy(out_pkt_p->type_data + offset,
	       plugin->username, plugin->username_length);
	offset += plugin->username_length;
	out_pkt_p->type_data[offset++] = '\0';
	memcpy(out_pkt_p->type_data + offset,
	       plugin->password, plugin->password_length);
    }
    else {
	memcpy(out_pkt_p->type_data, plugin->password, plugin->password_length);
    }
    return ((EAPPacketRef)out_pkt_p);
 failed:
    if (out_pkt_p != NULL) {
	free(out_pkt_p);
    }
    return (NULL);
}

static EAPClientStatus
eapgtc_init(EAPClientPluginDataRef plugin, 
	    CFArrayRef * required_props, 
	    EAPClientDomainSpecificError * error)
{
    *error = 0;
    *required_props = NULL;
    return (kEAPClientStatusOK);
}

static void
eapgtc_free(EAPClientPluginDataRef plugin)
{
    /* ignore, no context data */
    return;
}

static void
eapgtc_free_packet(EAPClientPluginDataRef plugin, EAPPacketRef arg)
{
    if (arg != NULL) {
	/* we malloc'd the packet, so free it */
	free(arg);
    }
    return;
}

static EAPClientState
eapgtc_process(EAPClientPluginDataRef plugin, 
	       const EAPPacketRef in_pkt,
	       EAPPacketRef * out_pkt_p,
	       EAPClientStatus * client_status,
	       EAPClientDomainSpecificError * error)
{
    EAPClientState	plugin_state;

    *client_status = kEAPClientStatusOK;
    *error = 0;
    plugin_state = kEAPClientStateAuthenticating;
    *out_pkt_p = NULL;

    switch (in_pkt->code) {
    case kEAPCodeRequest:
	if (plugin->password == NULL) {
	    *client_status = kEAPClientStatusUserInputRequired;
	}
	else {
	    *out_pkt_p = eapgtc_request(plugin,
					(const EAPRequestPacketRef)in_pkt);
	}
	break;
    case kEAPCodeSuccess:
	plugin_state = kEAPClientStateSuccess;
	break;
    case kEAPCodeFailure:
	*client_status = kEAPClientStatusFailed;
	plugin_state = kEAPClientStateFailure;
	break;
    default:
	break;
    }
    return (plugin_state);
}

static CFArrayRef
eapgtc_require_props(EAPClientPluginDataRef plugin)
{
    CFStringRef		prop;

    if (plugin->password != NULL) {
	return (NULL);
    }
    prop = kEAPClientPropUserPassword;
    return (CFArrayCreate(NULL, (const void **)&prop, 1,
			  &kCFTypeArrayCallBacks));
}

static EAPType 
eapgtc_type()
{
    return (kEAPTypeGenericTokenCard);
}

static const char *
eapgtc_name()
{
    return ("GTC");
}

static EAPClientPluginVersion 
eapgtc_version()
{
    return (kEAPClientPluginVersion);
}

static struct func_table_ent {
    const char *		name;
    void *			func;
} func_table[] = {
    { kEAPClientPluginFuncNameVersion, eapgtc_version },
    { kEAPClientPluginFuncNameEAPType, eapgtc_type },
    { kEAPClientPluginFuncNameEAPName, eapgtc_name },
    { kEAPClientPluginFuncNameInit, eapgtc_init },
    { kEAPClientPluginFuncNameFree, eapgtc_free },
    { kEAPClientPluginFuncNameProcess, eapgtc_process },
    { kEAPClientPluginFuncNameRequireProperties, eapgtc_require_props },
    { kEAPClientPluginFuncNameFreePacket, eapgtc_free_packet },
    { NULL, NULL},
};


EAPClientPluginFuncRef
eapgtc_introspect(EAPClientPluginFuncName name)
{
    struct func_table_ent * scan;

    for (scan = func_table; scan->name != NULL; scan++) {
	if (strcmp(name, scan->name) == 0) {
	    return (scan->func);
	}
    }
    return (NULL);
}
