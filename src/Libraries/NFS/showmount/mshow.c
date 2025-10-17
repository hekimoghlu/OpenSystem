/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/queue.h>

#include <CoreFoundation/CoreFoundation.h>
#include <dns_sd.h>

#include <nfs/rpcv2.h>
#include "showmount.h"

struct cbinfo {
	DNSServiceRef sdref;
	CFSocketRef sockref;
	CFRunLoopSourceRef rls;
};

struct cbinfo nfsinfo, mountdinfo;

/*
 * CFRunloop callback that calls DNSServiceProcessResult() when
 * there's new data on the socket.
 */
static void
socket_callback(CFSocketRef s, CFSocketCallBackType callbackType, CFDataRef address, const void *data, void *context)
{
	struct cbinfo *info = context;
	DNSServiceErrorType err;

	if (callbackType == kCFSocketNoCallBack) {
		printf("socket_callback: kCFSocketNoCallBack?\n");
		return;
	}

	if ((err = DNSServiceProcessResult(info->sdref)) != kDNSServiceErr_NoError) {
		printf("DNSServiceProcessResult() returned an error! %d\n", err);
		if (err == kDNSServiceErr_BadReference) {
			printf("bad reference?: %p, %d, %p, %p %p\n", s, (int)callbackType, address, data, context);
			return;
		}
		if ((context == &nfsinfo) || (context == &mountdinfo)) {
			/* bail if there's a problem with the main browse connection */
			exit(1);
		}
		/* dump the troublesome service connection */
		CFRunLoopRemoveSource(CFRunLoopGetCurrent(), info->rls, kCFRunLoopDefaultMode);
		CFRelease(info->rls);
		CFSocketInvalidate(info->sockref);
		CFRelease(info->sockref);
		DNSServiceRefDeallocate(info->sdref);
		free(info);
	}
}

#ifdef DEBUG
#define Dused
#else
#define Dused   __unused
#endif

static void
resolve_callback(
	__unused DNSServiceRef sdRef,
	__unused DNSServiceFlags flags,
	__unused uint32_t interfaceIndex,
	__unused DNSServiceErrorType errorCode,
	Dused const char *fullName,
	const char *hostTarget,
	Dused uint16_t port,
	Dused uint16_t txtLen,
	Dused const unsigned char *txtRecord,
	void *context)
{
	struct cbinfo *info = context;
#ifdef DEBUG
	const char *p;
	char *q, *s;
	int len;

	printf("resolve: %s %s:%d  TXT %d\n", fullName, hostTarget, port, txtLen);
	if ((txtLen > 1) && ((s = malloc(txtLen + 2)))) {
		p = txtRecord;
		q = s;
		len = txtLen;
		while (len > 0) {
			strncpy(q, p + 1, *p);
			len -= *p + 1;
			q += *p;
			p += *p + 1;
		}
		*q = '\0';
		printf("    %s\n", s);
		free(s);
	}
#endif
	do_print(hostTarget);

	CFRunLoopRemoveSource(CFRunLoopGetCurrent(), info->rls, kCFRunLoopDefaultMode);
	CFRelease(info->rls);
	CFSocketInvalidate(info->sockref);
	CFRelease(info->sockref);
	DNSServiceRefDeallocate(info->sdref);
	free(info);
}

/*
 * Handle newly-discovered services
 */
static void
browser_callback(
	__unused DNSServiceRef sdRef,
	DNSServiceFlags servFlags,
	uint32_t interfaceIndex,
	DNSServiceErrorType errorCode,
	const char *serviceName,
	const char *regType,
	const char *replyDomain,
	__unused void *context)
{
	DNSServiceErrorType err;
	CFSocketContext ctx = { 0, NULL, NULL, NULL, NULL };
	struct cbinfo *info;

	if (errorCode != kDNSServiceErr_NoError) {
		printf("DNS service discovery error: %d\n", errorCode);
		return;
	}
#ifdef DEBUG
	printf("browse: %s: %s, %s, %s\n",
	    (servFlags & kDNSServiceFlagsAdd) ? "new" : "gone",
	    serviceName, regType, replyDomain);
#endif
	if (!(servFlags & kDNSServiceFlagsAdd)) {
		return;
	}

	info = malloc(sizeof(*info));
	if (!info) {
		printf("browse: out of memeory\n");
		return;
	}

	err = DNSServiceResolve(&info->sdref, servFlags, interfaceIndex, serviceName, regType, replyDomain, resolve_callback, info);
	if (err != kDNSServiceErr_NoError) {
		printf("DNSServiceResolve failed: %d\n", err);
		free(info);
		return;
	}
	ctx.info = (void*)info;
	info->sockref = CFSocketCreateWithNative(kCFAllocatorDefault, DNSServiceRefSockFD(info->sdref),
	    kCFSocketReadCallBack, socket_callback, &ctx);
	if (!info->sockref) {
		printf("CFSocketCreateWithNative failed\n");
		DNSServiceRefDeallocate(info->sdref);
		free(info);
		return;
	}
	info->rls = CFSocketCreateRunLoopSource(kCFAllocatorDefault, info->sockref, 1);
	CFRunLoopAddSource(CFRunLoopGetCurrent(), info->rls, kCFRunLoopDefaultMode);
}

int
browse(void)
{
	DNSServiceErrorType err;
	CFSocketContext ctx = { 0, NULL, NULL, NULL, NULL };

	err = DNSServiceBrowse(&nfsinfo.sdref, 0, 0, "_nfs._tcp", NULL, browser_callback, NULL);
	if (err != kDNSServiceErr_NoError) {
		return 1;
	}
	ctx.info = (void*)&nfsinfo;
	nfsinfo.sockref = CFSocketCreateWithNative(kCFAllocatorDefault, DNSServiceRefSockFD(nfsinfo.sdref),
	    kCFSocketReadCallBack, socket_callback, &ctx);
	if (!nfsinfo.sockref) {
		return 1;
	}
	nfsinfo.rls = CFSocketCreateRunLoopSource(kCFAllocatorDefault, nfsinfo.sockref, 1);
	CFRunLoopAddSource(CFRunLoopGetCurrent(), nfsinfo.rls, kCFRunLoopDefaultMode);

	/* For backwards compatibility, browse for "mountd" services too */
	err = DNSServiceBrowse(&mountdinfo.sdref, 0, 0, "_mountd._tcp", NULL, browser_callback, NULL);
	if (err != kDNSServiceErr_NoError) {
		return 1;
	}
	ctx.info = (void*)&mountdinfo;
	mountdinfo.sockref = CFSocketCreateWithNative(kCFAllocatorDefault, DNSServiceRefSockFD(mountdinfo.sdref),
	    kCFSocketReadCallBack, socket_callback, &ctx);
	if (!mountdinfo.sockref) {
		return 1;
	}
	mountdinfo.rls = CFSocketCreateRunLoopSource(kCFAllocatorDefault, mountdinfo.sockref, 1);
	CFRunLoopAddSource(CFRunLoopGetCurrent(), mountdinfo.rls, kCFRunLoopDefaultMode);

	CFRunLoopRun();

	CFRelease(nfsinfo.rls);
	CFSocketInvalidate(nfsinfo.sockref);
	CFRelease(nfsinfo.sockref);
	DNSServiceRefDeallocate(nfsinfo.sdref);
	CFRelease(mountdinfo.rls);
	CFSocketInvalidate(mountdinfo.sockref);
	CFRelease(mountdinfo.sockref);
	DNSServiceRefDeallocate(mountdinfo.sdref);
	return 0;
}
