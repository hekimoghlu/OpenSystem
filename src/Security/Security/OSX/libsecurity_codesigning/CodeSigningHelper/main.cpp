/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#include <Security/CodeSigning.h>
#include <Security/SecCodePriv.h>
#include <xpc/xpc.h>
#include <sandbox.h>
#include <security_utilities/cfutilities.h>
#include <security_utilities/cfmunge.h>
#include <security_utilities/logging.h>
#include "codedirectory.h"



static void
request(xpc_connection_t peer, xpc_object_t event)
{
	pid_t pid = (pid_t)xpc_dictionary_get_int64(event, "pid");
	if (pid <= 0)
		return;
	
	size_t audit_size;
	audit_token_t const *audit =
		(audit_token_t const *)xpc_dictionary_get_data(event, "audit", &audit_size);
	
	if (audit != NULL && audit_size != sizeof(audit_token_t)) {
		Syslog::error("audit token has unexpected size %zu", audit_size);
		return;
	}
	
	xpc_object_t reply = xpc_dictionary_create_reply(event);
	if (reply == NULL)
		return;
	
	CFTemp<CFMutableDictionaryRef> attributes("{%O=%d}", kSecGuestAttributePid, pid);
    
	if (audit != NULL) {
		CFRef<CFDataRef> auditData = makeCFData(audit, audit_size);
		CFDictionaryAddValue(attributes.get(), kSecGuestAttributeAudit,
							 auditData);
	}
	CFRef<SecCodeRef> code;
	if (SecCodeCopyGuestWithAttributes(NULL, attributes, kSecCSDefaultFlags, &code.aref()) == noErr) {
		
		// path to base of client code
		CFRef<CFURLRef> codePath;
		if (SecCodeCopyPath(code, kSecCSDefaultFlags, &codePath.aref()) == noErr) {
			CFRef<CFDataRef> data = CFURLCreateData(NULL, codePath, kCFStringEncodingUTF8, true);
			xpc_dictionary_set_data(reply, "bundleURL", CFDataGetBytePtr(data), CFDataGetLength(data));
		}
		
		// if the caller wants the Info.plist, get it and verify the hash passed by the caller
		size_t iphLength;
		if (const void *iphash = xpc_dictionary_get_data(event, "infohash", &iphLength)) {
			if (CFRef<CFDataRef> data = SecCodeCopyComponent(code, Security::CodeSigning::cdInfoSlot, CFTempData(iphash, iphLength))) {
				xpc_dictionary_set_data(reply, "infoPlist", CFDataGetBytePtr(data), CFDataGetLength(data));
			}
		}
	}
	xpc_connection_send_message(peer, reply);
	xpc_release(reply);
}


static void CodeSigningHelper_peer_event_handler(xpc_connection_t peer, xpc_object_t event)
{
	xpc_type_t type = xpc_get_type(event);
	if (type == XPC_TYPE_ERROR)
		return;
	
	assert(type == XPC_TYPE_DICTIONARY);
	
	const char *cmd = xpc_dictionary_get_string(event, "command");
	if (cmd == NULL) {
		xpc_connection_cancel(peer);
	} else if (strcmp(cmd, "fetchData") == 0)
		request(peer, event);
	else {
		Syslog::error("peer sent invalid command %s", cmd);
		xpc_connection_cancel(peer);
	}
}


static void CodeSigningHelper_event_handler(xpc_connection_t peer)
{
	xpc_connection_set_event_handler(peer, ^(xpc_object_t event) {
		CodeSigningHelper_peer_event_handler(peer, event);
	});
	xpc_connection_resume(peer);
}

int main(int argc, const char *argv[])
{
	char *error = NULL;
	if (sandbox_init("com.apple.CodeSigningHelper", SANDBOX_NAMED, &error)) {
		Syslog::error("failed to enter sandbox: %s", error);
		exit(EXIT_FAILURE);
	}
	xpc_main(CodeSigningHelper_event_handler);
	return 0;
}
