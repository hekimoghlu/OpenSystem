/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#ifndef __DNSINFO_CREATE_H__
#define __DNSINFO_CREATE_H__

/*
 * These routines provide access to the systems DNS configuration
 */

#include <TargetConditionals.h>
#include <os/availability.h>
#include <sys/cdefs.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <dnsinfo.h>

typedef const struct __dns_create_config *      dns_create_config_t;
typedef const struct __dns_create_resolver *    dns_create_resolver_t;

#ifndef _PATH_RESOLVER_DIR
#if	TARGET_OS_OSX
#define _PATH_RESOLVER_DIR "/etc/resolver"
#else
#define _PATH_RESOLVER_DIR "/Library/Preferences/SystemConfiguration/resolver"
#endif
#endif

__BEGIN_DECLS

/*
 * DNS configuration creation APIs
 */
dns_create_config_t
_dns_configuration_create       (void)						API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_configuration_add_resolver (dns_create_config_t	*_config,
				 dns_create_resolver_t	_resolver)		API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_configuration_signature	(dns_create_config_t	*_config,
				 unsigned char		*signature,
				 size_t			signature_len)		API_AVAILABLE(macos(10.7), ios(5.0));	// signature_len >= CC_SHA256_DIGEST_LENGTH

void
_dns_configuration_free		(dns_create_config_t	*_config)		API_AVAILABLE(macos(10.4), ios(2.0));

/*
 * DNS [resolver] configuration creation APIs
 */
dns_create_resolver_t
_dns_resolver_create		(void)						API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_resolver_set_domain	(dns_create_resolver_t	*_resolver,
				 const char		*domain)		API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_resolver_add_nameserver	(dns_create_resolver_t	*_resolver,
				 struct sockaddr	*nameserver)		API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_resolver_add_search	(dns_create_resolver_t	*_resolver,
				 const char		*search)		API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_resolver_add_sortaddr	(dns_create_resolver_t	*_resolver,
				 dns_sortaddr_t		*sortaddr)		API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_resolver_set_configuration_identifier
				(dns_create_resolver_t	*_resolver,
				 const char		*config_identifier)	API_AVAILABLE(macos(10.11), ios(9.0));

void
_dns_resolver_set_flags		(dns_create_resolver_t	*_resolver,
				 uint32_t		flags)			API_AVAILABLE(macos(10.7), ios(4.0));

void
_dns_resolver_set_if_index	(dns_create_resolver_t	*_resolver,
				 uint32_t		if_index,
				 const char		*if_name)		API_AVAILABLE(macos(10.7), ios(4.0));

void
_dns_resolver_set_options	(dns_create_resolver_t	*_resolver,
				 const char		*options)		API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_resolver_set_order		(dns_create_resolver_t	*_resolver,
				 uint32_t		order)			API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_resolver_set_port		(dns_create_resolver_t	*_resolver,
				 uint16_t		port)			API_AVAILABLE(macos(10.4), ios(2.0));	// host byte order

void
_dns_resolver_set_timeout	(dns_create_resolver_t	*_resolver,
				 uint32_t		timeout)		API_AVAILABLE(macos(10.4), ios(2.0));

void
_dns_resolver_set_service_identifier
				(dns_create_resolver_t	*_resolver,
				 uint32_t		service_identifier)	API_AVAILABLE(macos(10.9), ios(7.0));

void
_dns_resolver_free		(dns_create_resolver_t	*_resolver)		API_AVAILABLE(macos(10.4), ios(2.0));

/*
 * DNS [resolver] flat-file configuration creation APIs
 */
void
_dnsinfo_flatfile_add_resolvers	(dns_create_config_t	*config)		API_AVAILABLE(macos(10.6)) SPI_AVAILABLE(ios(10.15), tvos(13.0), watchos(6.0), bridgeos(6.0));

void
_dnsinfo_flatfile_set_flags	(uint32_t		flags)			API_AVAILABLE(macos(10.9)) SPI_AVAILABLE(ios(10.15), tvos(13.0), watchos(6.0), bridgeos(6.0));

__END_DECLS

#endif	/* __DNSINFO_CREATE_H__ */
