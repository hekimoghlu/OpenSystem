/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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
#include <netsmb/netbios.h>
#include <netsmb/smb_lib.h>
#include <netsmb/nb_lib.h>


#define NBNS_FMT_ERR	0x01	/* Request was invalidly formatted */
#define NBNS_SRV_ERR	0x02	/* Problem with NBNS, connot process name */
#define NBNS_NME_ERR	0x03	/* No such name */
#define NBNS_IMP_ERR	0x04	/* Unsupport request */
#define NBNS_RFS_ERR	0x05	/* For policy reasons server will not register this name fron this host */
#define NBNS_ACT_ERR	0x06	/* Name is owned by another host */
#define NBNS_CFT_ERR	0x07	/* Name conflict error  */

static int nb_resolve_wins(CFArrayRef WINSAddresses, CFMutableArrayRef *addressArray)
{
	CFIndex ii, count = CFArrayGetCount(WINSAddresses);
	int  error = ENOMEM;
	
	for (ii = 0; ii < count; ii++) {
		CFStringRef winsString = CFArrayGetValueAtIndex(WINSAddresses, ii);
		char winsName[SMB_MAX_DNS_SRVNAMELEN+1];
		
		if (winsString == NULL) {
			continue;		
		}
		
		CFStringGetCString(winsString, winsName,  sizeof(winsName), kCFStringEncodingUTF8);
		error = resolvehost(winsName, addressArray, NULL, NBNS_UDP_PORT_137, TRUE, FALSE);
		if (error == 0) {
			break;
		}
		os_log_debug(OS_LOG_DEFAULT, "can't resolve WINS[%d] %s, syserr = %s",
					 (int)ii, winsName, strerror(error));
	}
	return error;
}

/*
 * Used for resolving NetBIOS names
 */
int nb_ctx_resolve(struct nb_ctx *ctx, CFArrayRef WINSAddresses)
{
	int error = 0;

	if (WINSAddresses == NULL) {
		ctx->nb_ns.sin_addr.s_addr = htonl(INADDR_BROADCAST);
		ctx->nb_ns.sin_port = htons(NBNS_UDP_PORT_137);
		ctx->nb_ns.sin_family = AF_INET;
		ctx->nb_ns.sin_len = sizeof(ctx->nb_ns);
	} else {
		CFMutableArrayRef addressArray = NULL;
		CFMutableDataRef addressData = NULL;
		struct connectAddress *conn = NULL;
		
		error = nb_resolve_wins(WINSAddresses, &addressArray);
		if (error) {
			return error;
		}
		/* 
		 * At this point we have at least one IPv4 sockaddr in outAddressArray 
		 * that we can use. May want to change this in the future to try all
		 * address.
		 */
		addressData = (CFMutableDataRef)CFArrayGetValueAtIndex(addressArray, 0);
		if (addressData)
			conn = (struct connectAddress *)((void *)CFDataGetMutableBytePtr(addressData));
			
		if (conn)
			memcpy(&ctx->nb_ns, &conn->addr, conn->addr.sa_len);
		else
			error = ENOMEM;
		CFRelease(addressArray);
	}
	return error;
}

/*
 * Convert NetBIOS name lookup errors to UNIX errors
 */
int nb_error_to_errno(int error)
{
	switch (error) {
	case NBNS_FMT_ERR:
		error = EINVAL;
		break;
	case NBNS_SRV_ERR: 
		error = EBUSY;
		break;
	case NBNS_NME_ERR: 
		error = ENOENT;
		break;
	case NBNS_IMP_ERR: 
		error = ENOTSUP;
		break;
	case NBNS_RFS_ERR: 
		error = EACCES;
		break;
	case NBNS_ACT_ERR: 
		error = EADDRINUSE;
		break;
	case NBNS_CFT_ERR: 
		error = EADDRINUSE;
		break;
	default:
		error = ETIMEDOUT;
		break;
	};
	return error;
}
