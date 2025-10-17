/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#ifndef _S_BOOTPLIB_SUBNETS_H
#define _S_BOOTPLIB_SUBNETS_H

/*
 * subnets.h
 * - API's to access DHCP server subnet information
 */

/*
 * Modification History:
 * 
 * June 23, 2006	Dieter Siegmund (dieter@apple.com)
 * - initial revision (based on subnetDescr.h)
 */

#include <stdbool.h>
#include <netinet/in.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFArray.h>

#include "dhcp.h"

#define SUBNET_PROP__CREATOR		"_creator"
#define SUBNET_PROP_NAME		"name"
#define SUBNET_PROP_NET_ADDRESS		"net_address"
#define SUBNET_PROP_NET_MASK		"net_mask"
#define SUBNET_PROP_NET_RANGE		"net_range"
#define SUBNET_PROP_CLIENT_TYPES	"client_types"
#define SUBNET_PROP_SUPERNET		"supernet"
#define SUBNET_PROP_LEASE_MIN		"lease_min"
#define SUBNET_PROP_LEASE_MAX		"lease_max"


typedef bool (SubnetIsAddressInUseFunc)(void * private, struct in_addr ip);
typedef SubnetIsAddressInUseFunc * SubnetIsAddressInUseFuncRef;
typedef struct _SubnetList * SubnetListRef;
typedef struct _Subnet * SubnetRef;

/**
 ** SubnetListRef API's
 **/

SubnetRef
SubnetListAcquireAddress(SubnetListRef list, struct in_addr * addr,
			 SubnetIsAddressInUseFuncRef func, void * arg);

SubnetRef
SubnetListGetSubnetForAddress(SubnetListRef list, struct in_addr addr,
			      bool in_range);

bool
SubnetListAreAddressesOnSameSupernet(SubnetListRef list,
				     struct in_addr addr,
				     struct in_addr other_addr);

SubnetListRef
SubnetListCreateWithArray(CFArrayRef list);

void
SubnetListFree(SubnetListRef * subnets);

void
SubnetListPrint(SubnetListRef subnets);

void
SubnetListPrintCFString(CFMutableStringRef str, SubnetListRef subnets);

/**
 ** SubnetRef API's
 **/
dhcp_lease_time_t
SubnetGetMaxLease(SubnetRef subnet);

dhcp_lease_time_t
SubnetGetMinLease(SubnetRef subnet);

const char *
SubnetGetOptionPtrAndLength(SubnetRef subnet, dhcptag_t tag,
			    int * option_length);

struct in_addr
SubnetGetMask(SubnetRef subnet);

bool
SubnetDoesAllocate(SubnetRef subnet);

#endif /* _S_BOOTPLIB_SUBNETS_H */
