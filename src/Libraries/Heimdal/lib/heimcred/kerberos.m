/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#import <TargetConditionals.h>

#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import <CoreFoundation/CFRuntime.h>

#import "heimcred.h"
#import "heimbase.h"
#import "common.h"
#import "gsscred.h"
/*
 *
 */

void
_HeimCredRegisterKerberos(void)
{
    CFMutableSetRef set = CFSetCreateMutable(NULL, 0, &kCFTypeSetCallBacks);
    CFMutableDictionaryRef schema;

    schema = _HeimCredCreateBaseSchema(kHEIMObjectKerberos);

    CFDictionarySetValue(schema, kHEIMAttrTransient, CFSTR("b"));
    CFDictionarySetValue(schema, kHEIMAttrStatus, CFSTR("n"));
    CFDictionarySetValue(schema, kHEIMAttrAuthTime, CFSTR("t"));
    CFDictionarySetValue(schema, kHEIMAttrExpire, CFSTR("t"));
    CFDictionarySetValue(schema, kHEIMAttrRenewTill, CFSTR("t"));
    CFDictionarySetValue(schema, kHEIMAttrKerberosTicketGrantingTicket, CFSTR("b"));

    CFSetAddValue(set, schema);
    CFRELEASE_NULL(schema);


    _HeimCredRegisterMech(kHEIMTypeKerberos, set, KerberosStatusCallback, NULL, HeimCredGlobalCTX.notifyCaches, DefaultTraceCallback, false, NULL);
    CFRELEASE_NULL(set);
}
