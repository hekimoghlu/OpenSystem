/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCValidation.h>
#include <SystemConfiguration/SCDynamicStorePrivate.h>
#include <CoreFoundation/CoreFoundation.h>
#include "PMStore.h"
#include "PrivateLib.h"


/* TBD
typedef (void *)PMStoreKeysChangedCallBack(void *param, CFArrayRef keys);
__private_extern__ void PMStoreRequestCallBack(void *param, (PMStoreKeysChangedCallBack *)callback, CFArrayRef keys);
*/

static CFMutableDictionaryRef   gPMStore = NULL;
SCDynamicStoreRef               gSCDynamicStore = NULL;

static void PMDynamicStoreDisconnectCallBack(SCDynamicStoreRef store, void *info __unused);

/* dynamicStoreNotifyCallBack
 * defined in pmconfigd.c
 */
__private_extern__ void dynamicStoreNotifyCallBack(
    SCDynamicStoreRef   store,
    CFArrayRef          changedKeys,
    void                *info);


void PMStoreLoad(void)
{
    gPMStore = CFDictionaryCreateMutable(0, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

    gSCDynamicStore = SCDynamicStoreCreate(0, CFSTR("powerd"), dynamicStoreNotifyCallBack, NULL);

    if (gSCDynamicStore) {
        SCDynamicStoreSetDispatchQueue(gSCDynamicStore, _getPMMainQueue());
    }

    SCDynamicStoreSetDisconnectCallBack(gSCDynamicStore, PMDynamicStoreDisconnectCallBack);
}

bool PMStoreSetValue(CFStringRef key, CFTypeRef value)
{
    CFTypeRef lastValue = NULL;

    if (!key || !value || !gPMStore)
        return false;

    if (!isA_CFString(key)) {
        return false;
    }

    lastValue = CFDictionaryGetValue(gPMStore, key);

    if (lastValue && CFEqual(lastValue, value)) {
        return true;
    }

    CFDictionarySetValue(gPMStore, key, value);
    return SCDynamicStoreSetValue(gSCDynamicStore, key, value);
}

bool PMStoreRemoveValue(CFStringRef key)
{
    if (!key || !isA_CFString(key)) {
        return false;
    }

    CFDictionaryRemoveValue(gPMStore, key);
    return SCDynamicStoreRemoveValue(gSCDynamicStore, key);
}

static void PMDynamicStoreDisconnectCallBack(
    SCDynamicStoreRef           store,
    void                        *info __unused)
{
    assert (store == gSCDynamicStore);

    SCDynamicStoreSetMultiple(gSCDynamicStore, gPMStore, NULL, NULL);
}
