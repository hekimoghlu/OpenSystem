/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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
#include "dns_sd.h"
#include "dnssd_ipc.h"

DNSServiceErrorType
DNSServiceBrowseInternal(DNSServiceRef *sdRef, DNSServiceFlags flags, uint32_t interfaceIndex, const char *regtype,
    const char *domain, const DNSServiceAttribute *attr, DNSServiceBrowseReply callBack, void *context);

DNSServiceErrorType
DNSServiceResolveInternal(DNSServiceRef *sdRef, DNSServiceFlags flags, uint32_t interfaceIndex, const char *name,
    const char *regtype, const char *domain, const DNSServiceAttribute *attr, DNSServiceResolveReply callBack,
    void *context);

DNSServiceErrorType
DNSServiceGetAddrInfoInternal(DNSServiceRef *sdRef, DNSServiceFlags flags, uint32_t interfaceIndex,
    DNSServiceProtocol protocol, const char *hostname, const DNSServiceAttribute *attr, DNSServiceGetAddrInfoReply callBack,
    void *context);

DNSServiceErrorType
DNSServiceQueryRecordInternal(DNSServiceRef *sdRef, DNSServiceFlags flags, uint32_t interfaceIndex, const char *name,
    uint16_t rrtype, uint16_t rrclass, const DNSServiceAttribute *attr, const DNSServiceQueryRecordReply callback,
    void *context);

DNSServiceErrorType
DNSServiceRegisterInternal(DNSServiceRef *sdRef, DNSServiceFlags flags, uint32_t interfaceIndex, const char *name,
    const char *regtype, const char *domain, const char *host, uint16_t portInNetworkByteOrder, uint16_t txtLen,
    const void *txtRecord, const DNSServiceAttribute *attr, DNSServiceRegisterReply callBack, void *context);

DNSServiceErrorType
DNSServiceRegisterRecordInternal(DNSServiceRef sdRef, DNSRecordRef *recordRef, DNSServiceFlags flags,
    uint32_t interfaceIndex, const char *fullname, uint16_t rrtype, uint16_t rrclass, uint16_t rdlen,
    const void *rdata, uint32_t ttl, const DNSServiceAttribute *attr, DNSServiceRegisterRecordReply callBack,
    void *context);

DNSServiceErrorType
DNSServiceSendQueuedRequestsInternal(DNSServiceRef sdr);
