/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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

mDNSexport mStatus handle_client_request_ut(void *req)
{
	return handle_client_request((request_state*)req);
}

mDNSexport void LogCacheRecords_ut(mDNSs32 now, mDNSu32* retCacheUsed, mDNSu32* retCacheActive)
{
	mDNSu32 CacheUsed =0, CacheActive =0, slot;
	const CacheGroup *cg;
	const CacheRecord *cr;

	LogMsgNoIdent("------------ Cache -------------");
	LogMsgNoIdent("Slt Q     TTL if     U Type rdlen");
	for (slot = 0; slot < CACHE_HASH_SLOTS; slot++)
	{
		for (cg = mDNSStorage.rrcache_hash[slot]; cg; cg=cg->next)
		{
			CacheUsed++;    // Count one cache entity for the CacheGroup object
			for (cr = cg->members; cr; cr=cr->next)
			{
				const mDNSs32 remain = cr->resrec.rroriginalttl - (now - cr->TimeRcvd) / mDNSPlatformOneSecond;
				const char *ifname;
				mDNSInterfaceID InterfaceID = cr->resrec.InterfaceID;
				if (!InterfaceID && cr->resrec.rDNSServer && (cr->resrec.rDNSServer->scopeType != kScopeNone))
					InterfaceID = cr->resrec.rDNSServer->interface;
				ifname = InterfaceNameForID(&mDNSStorage, InterfaceID);
				if (cr->CRActiveQuestion) CacheActive++;
				PrintOneCacheRecordToFD(STDOUT_FILENO, cr, slot, remain, ifname, &CacheUsed);
				PrintCachedRecordsToFD(STDOUT_FILENO, cr, slot, remain, ifname, &CacheUsed);
			}
		}
	}

	*retCacheUsed = CacheUsed;
	*retCacheActive = CacheActive;
}

mDNSexport int LogEtcHosts_ut(mDNS *const m)
{
	return LogEtcHostsToFD(STDOUT_FILENO, m);
}
