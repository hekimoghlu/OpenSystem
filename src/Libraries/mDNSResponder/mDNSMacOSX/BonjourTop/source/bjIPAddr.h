/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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

//
//  bjIPAddr.h
//  TestTB
//
//  Created by Terrin Eager on 1/19/13.
//
//

#ifndef __TestTB__bjIPAddr__
#define __TestTB__bjIPAddr__

#include <iostream>
#include <sys/socket.h>
#include "bjtypes.h"

class BJIPAddr
{
public:
    BJIPAddr();
    BJIPAddr(const BJIPAddr& src);
    BJIPAddr &operator=(const BJIPAddr& src);

    void Empty();

    bool IsBonjourMulticast();
    bool IsSameSubNet(BJIPAddr* addr);

    bool IsIPv4();
    bool IsIPv6();
    bool IsIPv6LinkLocal();
    bool IsEmpty();
    bool IsEmptySubnet();

    void Set(const char* addr);
    void Setv6(const char* addr);
    void Set(struct in6_addr* ipi6_addr);
    void Set(struct in_addr* ip_addr);
    void Set(struct sockaddr_storage* sockStorage);
    void Setv4Raw(BJ_UINT8* ipi4_addr);
    void Setv6Raw(BJ_UINT8* ipi6_addr);

    sockaddr_storage* GetRawValue();
    struct in6_addr* Getin6_addr();

    void CreateLinkLocalIPv6(BJ_UINT8* mac);
    BJ_COMPARE Compare(BJIPAddr* addr);
    BJ_UINT16 GetPortNumber();
    char* GetString();
private:
    sockaddr_storage sockAddrStorage;
    BJ_INT32 IPv4SubNet;
    char stringbuffer[100];
    static sockaddr_storage emptySockAddrStorage;
};


#endif /* defined(__TestTB__bjIPAddr__) */
