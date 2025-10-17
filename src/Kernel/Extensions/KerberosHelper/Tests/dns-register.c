/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <dns_sd.h>
#include <err.h>
#include <unistd.h>


void
TXTRegisterCallback(DNSServiceRef sdRef __attribute__((unused)),
		    DNSRecordRef RecordRef __attribute__((unused)), 
		    DNSServiceFlags flags __attribute__((unused)), 
		    DNSServiceErrorType errorCode __attribute__((unused)),
		    void *context __attribute__((unused)))
{
}


int
main(int argc __attribute__((unused)), char **argv __attribute__((unused)))
{
    DNSServiceErrorType error;
    DNSServiceRef dnsRef;
    int i, fd;

    if (argc < 2)
	errx(1, "argc < 2");

    for (i = 1; i < argc; i++) {
	DNSRecordRef recordRef;
	char *hostname = argv[i];
	char *recordName;
	char *realm;
	size_t len;
	const char *prefix = "LKDC:SHA1.fake";

	asprintf(&recordName, "_kerberos.%s.", hostname);
	if (recordName == NULL)
	    errx(1, "malloc");
	
	len = strlen(prefix) + strlen(hostname);
	asprintf(&realm, "%c%s%s", (int)len, prefix, hostname);
	if (realm == NULL)
	    errx(1, "malloc");
	
	error = DNSServiceCreateConnection(&dnsRef);
	if (error)
	    errx(1, "DNSServiceCreateConnection");
	
	error =  DNSServiceRegisterRecord(dnsRef, 
					  &recordRef,
					  kDNSServiceFlagsShared | kDNSServiceFlagsAllowRemoteQuery,
					  0,
					  recordName,
					  kDNSServiceType_TXT,
					  kDNSServiceClass_IN,
					  len+1,
					  realm,
					  300,
					  TXTRegisterCallback,
					  NULL);
	if (error)
	    errx(1, "DNSServiceRegisterRecord: %d", error);
    }

    fd = DNSServiceRefSockFD(dnsRef);

    while (1) {
	int ret;
	fd_set rfd;

	FD_ZERO(&rfd);
	FD_SET(fd, &rfd);

	ret = select(fd + 1, &rfd, NULL, NULL, NULL);
	if (ret == 0)
	    errx(1, "timeout ?");
	else if (ret < 0)
	    err(1, "select");
	
	if (FD_ISSET(fd, &rfd))
	    DNSServiceProcessResult(dnsRef);
    }
}
