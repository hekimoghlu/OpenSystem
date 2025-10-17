/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#ifndef __DNSINFO_PRIVATE_H__
#define __DNSINFO_PRIVATE_H__


#include <os/availability.h>
#include <sys/cdefs.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <mach/mach.h>

#include "dnsinfo.h"

/*
 * What's all of this stuff about???
 *
 * In order to minimize the amount of data conversions associated with
 * storing, retrieving, and accessing the systems DNS configuration I
 * have opted to create a memory image filled with the actual data
 * types / structures that would be used by those applications which
 * make use of these APIS.  The implication here is that we use
 * sockaddr's for addresses, char *'s for strings, etc.
 *
 * To achieve this goal the APIs which are used to store and access the
 * configuration data build and use a single buffer of the relevant
 * bits.  When multiple instances of a given type / structure are needed
 * we maintain a count, the actual data, and when the configuration has
 * been unpacked, arrays of pointers to the data.
 *
 * In memory, the data looks as follows:
 *
 * +-------------------------------------------------------------------+
 * | struct _dns_config_buf_t                                          |
 * +-+-------------+---------------------------------------------------+
 * | | config      | struct dns_config_t                               |
 * | |             +-+-----------------------------+-------------------+
 * | |             | | n_resolver                  | int32_t           | <- # of name resolvers
 * | |             | +-----------------------------+-------------------+
 * | |             | | resolver                    | dns_resolver_t ** | <- not used during creation, filled
 * | |             | |                             |                   |    in with pointer to a list of
 * | |             | |                             |                   |    resolver configurations that will
 * | |             | |                             |                   |    be established in the "padding"
 * | |             | +-----------------------------+--------------------+
 * | |             | | n_scoped_resolver           | int32_t           | <- # of name scoped resolvers
 * | |             | +-----------------------------+--------------------+
 * | |             | | scoped_resolver             | dns_resolver_t ** | <- not used during creation, filled
 * | |             | |                             |                   |    in with pointer to a list of scoped
 * | |             | |                             |                   |    resolver configurations  that will
 * | |             | |                             |                   |    be established in the "padding"
 * | |             | +-----------------------------+--------------------+
 * | |             | | generation                  | uint64_t          | <- generation # of configuration
 * | |             | +-----------------------------+-------------------+
 * | |             | | n_service_specific_resolver | int32_t           | <- # of name service-specific resolvers
 * | |             | +-------------------+---------+-------------------+
 * | |             | | service_specific_resolver   | dns_resolver_t ** | <- not used during creation, filled
 * | |             | |                             |                   |    in with pointer to a list of service-specific
 * | |             | |                             |                   |    resolver configurations that will be
 * | |             | |                             |                   |    established in the "padding"
 * | |             | +-----------------------------+-------------------+
 * | |             | | version                     | uint32_t          | <- configuration structure(s) version
 * | +-------------+-+-----------------------------+-------------------+
 * | | n_attribute | uint32_t                                          | <- how many bytes of "attribute"
 * | |             |                                                   |    data is associated with the
 * | |             |                                                   |    configuration
 * |-+-------------+---------------------------------------------------+
 * | | n_padding   | uint32_t                                          | <- how many additional bytes
 * | |             |                                                   |    for arrays (of pointers), ...
 * +-+-------------+---------------------------------------------------+
 * | struct dns_attribute_t                                            |
 * |-+-------------+---------------------------------------------------+
 * | | type        | uint32_t                                          | <- type of attribute (e.g. CONFIG_ATTRIBUTE_RESOLVER)
 * | +-------------+---------------------------------------------------+
 * | | length      | uint32_t                                          | <- length of the attribute
 * | +-------------+---------------------------------------------------+
 * | | attribute   | struct _dns_resolver_buf_t                        | <- the attribute data (resolver configuration #1)
 * | |             +-+-------------+-----------------------------------+
 * | |             | | resolver    | struct dns_resolver_t             |
 * | |             | |             +--------------+--------------------+
 * | |             | |             | domain       | char *             | <- not used during creation,
 * | |             | |             |              |                    |    filled in with pointer to
 * | |             | |             |              |                    |    domain name in the "padding"
 * | |             | |             +--------------+--------------------+
 * | |             | |             | n_nameserver | int32_t            | <- # of name server addresses
 * | |             | |             +--------------+--------------------+
 * | |             | |             | nameserver   | struct sockaddr ** | <- not used during creation,
 * | |             | |             |              |                    |    filled in with pointer to
 * | |             | |             |              |                    |    a list of addresses which
 * | |             | |             |              |                    |    will be established in the
 * | |             | |             |              |                    |    "padding"
 * | |             | |             +--------------+--------------------+
 * | |             | |             | ...                               |
 * | |             +-+-------------+--------------+--------------------+
 * | |             | | n_attribute | uint32_t                          |
 * | |             +-+-------------+-----------------------------------+
 * | |             | | attribute   | struct dns_attribute_t            |
 * | |             | |             +-+-----------+---------------------+
 * | |             | |             | | type      | uint32_t            | <- type of attribute (e.g. RESOLVER_ATTRIBUTE_DOMAIN)
 * | |             | |             | +-----------+---------------------+
 * | |             | |             | | length    | uint32_t            | <- length of the attribute
 * | |             | |             | +-----------+---------------------+
 * | |             | |             | | attribute |                     | <- the attribute data ("apple.com")
 * | |             +-+-------------+-------------+---------------------+
 * | |             | | attribute   | struct dns_attribute_t            |
 * | |             | |             +-+-----------+---------------------+
 * | |             | |             | | type      | uint32_t            | <- type of attribute (e.g. RESOLVER_ATTRIBUTE_ADDRESS)
 * | |             | |             | +-----------+---------------------+
 * | |             | |             | | length    | uint32_t            | <- length of the attribute
 * | |             | |             | +-----------+---------------------+
 * | |             | |             | | attribute |                     | <- the attribute data ("struct sockaddr_in" #1)
 * | |             +---------------+-----------------------------------+
 * | |             | | attribute   | struct dns_attribute_t            |
 * | |             | |             +-+-----------+---------------------+
 * | |             | |             | | type      | uint32_t            | <- type of attribute (e.g. RESOLVER_ATTRIBUTE_ADDRESS)
 * | |             | |             | +-----------+---------------------+
 * | |             | |             | | length    | uint32_t            | <- length of the attribute
 * | |             | |             | +-----------+---------------------+
 * | |             | |             | | attribute |                     | <- the attribute data ("struct sockaddr_in" #2)
 * | |             +---------------+-----------------------------------+
 * | |             | ...                                               |
 * +-+-------------+---------------------------------------------------+
 * | | attribute   | struct _dns_resolver_buf_t                        | <- the attribute data (resolver configuration #2)
 * | |             +---------------+-----------------------------------+
 * | |             | ...                                               |
 * +---------------+---------------------------------------------------+
 * | | ...                                                             |
 * +---------------+---------------------------------------------------+
 *
 * When the data is unpacked the "n_padding" additional bytes
 * specified in configuration buffer will be allocated at the
 * end of this buffer. Arrays of pointers will be allocated
 * within the extra space array element (an "attribute") is
 * encountered the pointer reference will be filled in.
 */


// configuration buffer attributes
enum {
	CONFIG_ATTRIBUTE_RESOLVER	= 1,
	CONFIG_ATTRIBUTE_SCOPED_RESOLVER,
	CONFIG_ATTRIBUTE_SERVICE_SPECIFIC_RESOLVER,
};


// resolver buffer attributes
enum {
	RESOLVER_ATTRIBUTE_DOMAIN	= 10,
	RESOLVER_ATTRIBUTE_ADDRESS,
	RESOLVER_ATTRIBUTE_SEARCH,
	RESOLVER_ATTRIBUTE_SORTADDR,
	RESOLVER_ATTRIBUTE_OPTIONS,
	RESOLVER_ATTRIBUTE_CONFIGURATION_ID,
	RESOLVER_ATTRIBUTE_INTERFACE_NAME,
};


#pragma pack(4)
typedef struct {
	uint32_t	type;
	uint32_t	length;
	uint8_t		attribute[0];
} dns_attribute_t;
#pragma pack()


#pragma pack(4)
typedef struct {
	dns_config_t	config;
	uint32_t	n_attribute;
	uint32_t	n_padding;
	uint8_t		attribute[0];
} _dns_config_buf_t;
#pragma pack()


#pragma pack(4)
typedef struct {
	dns_resolver_t	resolver;
	uint32_t	n_attribute;
	uint8_t		attribute[0];
} _dns_resolver_buf_t;
#pragma pack()


__BEGIN_DECLS

__END_DECLS

#endif	/* __DNSINFO_PRIVATE_H__ */
