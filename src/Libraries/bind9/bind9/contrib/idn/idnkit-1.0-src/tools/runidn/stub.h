/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#ifndef STUB_H
#define STUB_H

#ifdef HAVE_GETHOSTBYNAME
extern struct hostent *
idn_stub_gethostbyname(const char *name);
#endif

#ifdef GETHOST_R_GLIBC_FLAVOR
#ifdef HAVE_GETHOSTBYNAME_R
extern int
idn_stub_gethostbyname_r(const char *name, struct hostent *result,
			 char *buffer, size_t buflen,
			 struct hostent **rp, int *errp);
#endif
#else /* GETHOST_R_GLIBC_FLAVOR */
#ifdef HAVE_GETHOSTBYNAME_R
extern struct hostent *
idn_stub_gethostbyname_r(const char *name, struct hostent *result,
			 char *buffer, int buflen, int *errp);
#endif
#endif /* GETHOST_R_GLIBC_FLAVOR */

#ifdef HAVE_GETHOSTBYNAME2
extern struct hostent *
idn_stub_gethostbyname2(const char *name, int af);
#endif

#ifdef GETHOST_R_GLIBC_FLAVOR
#ifdef HAVE_GETHOSTBYNAME2_R
extern int
idn_stub_gethostbyname2_r(const char *name, int af, struct hostent *result,
			  char *buffer, size_t buflen,
			  struct hostent **rp, int *errp);
#endif
#endif /* GETHOST_R_GLIBC_FLAVOR */

#ifdef HAVE_GETHOSTBYADDR
extern struct hostent *
idn_stub_gethostbyaddr(GHBA_ADDR_T addr, GHBA_ADDRLEN_T len, int type);
#endif

#ifdef GETHOST_R_GLIBC_FLAVOR
#ifdef HAVE_GETHOSTBYADDR_R
extern int
idn_stub_gethostbyaddr_r(GHBA_ADDR_T addr, GHBA_ADDRLEN_T len, int type,
			 struct hostent *result, char *buffer,
			 size_t buflen, struct hostent **rp, int *errp);
#endif
#else /* GETHOST_R_GLIBC_FLAVOR */
#ifdef HAVE_GETHOSTBYADDR_R
extern struct hostent *
idn_stub_gethostbyaddr_r(GHBA_ADDR_T addr, GHBA_ADDRLEN_T len, int type,
			 struct hostent *result, char *buffer,
			 int buflen, int *errp);
#endif
#endif /* GETHOST_R_GLIBC_FLAVOR */

#ifdef HAVE_GETIPNODEBYNAME
extern struct hostent *
idn_stub_getipnodebyname(const char *name, int af, int flags, int *errp);
#endif

#ifdef HAVE_GETIPNODEBYADDR
extern struct hostent *
idn_stub_getipnodebyaddr(const void *src, size_t len, int af, int *errp);
#endif

#ifdef HAVE_FREEHOSTENT
extern void
idn_stub_freehostent(struct hostent *hp);
#endif

#ifdef HAVE_GETADDRINFO
extern int
idn_stub_getaddrinfo(const char *nodename, const char *servname,
		     const struct addrinfo *hints, struct addrinfo **res);
#endif

#ifdef HAVE_FREEADDRINFO
extern void
idn_stub_freeaddrinfo(struct addrinfo *aip);
#endif

#ifdef HAVE_GETNAMEINFO
extern int
idn_stub_getnameinfo(const struct sockaddr *sa, GNI_SALEN_T salen,
		     char *host, GNI_HOSTLEN_T hostlen, char *serv,
		     GNI_SERVLEN_T servlen, GNI_FLAGS_T flags);
#endif

#endif /* STUB_H */
