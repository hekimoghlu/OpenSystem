/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
/*
 * DHCPDUIDIAID.h
 * - routines to set/access the DHCP client DUID and the IAIDs for particular
 *   interfaces
 */

#ifndef _S_DHCPDUIDIAID_H
#define _S_DHCPDUIDIAID_H


/* 
 * Modification History
 *
 * May 14, 2010
 * - created
 */

#include <CoreFoundation/CFData.h>
#include <stdint.h>
#include <stdbool.h>
#include "DHCPDUID.h"
#include "symbol_scope.h"
#include "interfaces.h"
#include "ipconfig_types.h"

PRIVATE_EXTERN CFDataRef
DHCPDUIDGet(void);

PRIVATE_EXTERN CFDataRef
DHCPDUIDEstablishAndGet(DHCPDUIDType type);

PRIVATE_EXTERN DHCPIAID
DHCPIAIDGet(const char * ifname);

PRIVATE_EXTERN CFDataRef
DHCPDUIDCopy(interface_t * if_p);

#endif /* _S_DHCPDUIDIAID_H */
