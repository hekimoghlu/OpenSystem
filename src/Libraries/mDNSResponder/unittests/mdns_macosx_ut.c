/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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

#include "DNSCommon.h"                  // Defines general DNS utility routines
#include "unittest_common.h"
#include "mDNSMacOSX.h"

// To match *either* a v4 or v6 instance of this interface
mDNSlocal mDNSInterfaceID SearchForInterfaceByAddr(mDNSAddr* addr)
{
	NetworkInterfaceInfoOSX *i;
	for (i = mDNSStorage.p->InterfaceList; i; i = i->next)
		if (i->Exists)
		{
			if ((i->ifinfo.ip.type == mDNSAddrType_IPv4) &&
				i->ifinfo.ip.ip.v4.NotAnInteger == addr->ip.v4.NotAnInteger)
				return i->ifinfo.InterfaceID;
			else if ((i->ifinfo.ip.type == mDNSAddrType_IPv6) &&
					 (i->ifinfo.ip.ip.v6.l[0] == addr->ip.v6.l[0] &&
					  i->ifinfo.ip.ip.v6.l[1] == addr->ip.v6.l[1] &&
					  i->ifinfo.ip.ip.v6.l[2] == addr->ip.v6.l[2] &&
					  i->ifinfo.ip.ip.v6.l[3] == addr->ip.v6.l[3])
					 )
				return i->ifinfo.InterfaceID;
		}
	return(NULL);
}

mDNSexport void SetInterfaces_ut(mDNSInterfaceID* pri_id, mDNSAddr *pri_v4, mDNSAddr* pri_v6, mDNSAddr* pri_router)
{
	mDNSs32 utc = mDNSPlatformUTC();

	MarkAllInterfacesInactive(utc);
	UpdateInterfaceList(utc);
	ClearInactiveInterfaces(utc);
	SetupActiveInterfaces(utc);

	// set primary interface info
	{
		mDNSAddr* addr;
		NetworkChangedKey_IPv4         = SCDynamicStoreKeyCreateNetworkGlobalEntity(NULL, kSCDynamicStoreDomainState, kSCEntNetIPv4);
		NetworkChangedKey_IPv6         = SCDynamicStoreKeyCreateNetworkGlobalEntity(NULL, kSCDynamicStoreDomainState, kSCEntNetIPv6);
		NetworkChangedKey_Hostnames    = SCDynamicStoreKeyCreateHostNames(NULL);
		NetworkChangedKey_Computername = SCDynamicStoreKeyCreateComputerName(NULL);
		NetworkChangedKey_DNS          = SCDynamicStoreKeyCreateNetworkGlobalEntity(NULL, kSCDynamicStoreDomainState, kSCEntNetDNS);
		NetworkChangedKey_StateInterfacePrefix = SCDynamicStoreKeyCreateNetworkInterfaceEntity(NULL, kSCDynamicStoreDomainState, CFSTR(""), NULL);

		mDNSPlatformGetPrimaryInterface(pri_v4, pri_v6, pri_router);
		addr = (pri_v4->type == mDNSAddrType_IPv4) ? pri_v4 : pri_v6;
		*pri_id = SearchForInterfaceByAddr(addr);

        MDNS_DISPOSE_CF_OBJECT(NetworkChangedKey_IPv4);
        MDNS_DISPOSE_CF_OBJECT(NetworkChangedKey_IPv6);
        MDNS_DISPOSE_CF_OBJECT(NetworkChangedKey_Hostnames);
        MDNS_DISPOSE_CF_OBJECT(NetworkChangedKey_Computername);
        MDNS_DISPOSE_CF_OBJECT(NetworkChangedKey_DNS);
        MDNS_DISPOSE_CF_OBJECT(NetworkChangedKey_StateInterfacePrefix);
	}
}

mDNSexport mDNSBool mDNSMacOSXCreateEtcHostsEntry_ut(const domainname *domain, const struct sockaddr *sa, const domainname *cname, char *ifname, AuthHash *auth)
{
	return mDNSMacOSXCreateEtcHostsEntry(domain, sa, cname, ifname, auth);
}

mDNSexport void UpdateEtcHosts_ut(void *context)
{
	mDNS_Lock(&mDNSStorage);
	UpdateEtcHosts(&mDNSStorage, context);
	mDNS_Unlock(&mDNSStorage);
}

mDNSexport void mDNSDomainLabelFromCFString_ut(CFStringRef cfs, domainlabel *const namelabel)
{
    mDNSDomainLabelFromCFString(cfs, namelabel);
}

mDNSexport mDNSu32 IndexForInterfaceByName_ut(const char *ifname)
{
    NetworkInterfaceInfoOSX * i = SearchForInterfaceByName(ifname, AF_UNSPEC);
    return (i ? i->scope_id : 0);
}
