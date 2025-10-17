/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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
/*!
 @header SOSViews.h - views
 */

#ifndef _sec_SOSViews_
#define _sec_SOSViews_

#include <CoreFoundation/CFRuntime.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecureObjectSync/SOSPeerInfo.h>
#include <xpc/xpc.h>

__BEGIN_DECLS

// Internal only views, do not export.
extern const CFStringRef kSOSViewKeychainV0;
extern const CFStringRef kSOSViewKeychainV0_tomb;
extern const CFStringRef kSOSViewBackupBagV0_tomb;
extern const CFStringRef kSOSViewWiFi_tomb;
extern const CFStringRef kSOSViewAutofillPasswords_tomb;
extern const CFStringRef kSOSViewSafariCreditCards_tomb;
extern const CFStringRef kSOSViewiCloudIdentity_tomb;
extern const CFStringRef kSOSViewOtherSyncable_tomb;

typedef struct __OpaqueSOSView {
    CFRuntimeBase _base;
    CFStringRef label;
    CFMutableDictionaryRef ringnames;
} *SOSViewRef;


typedef enum {
    kViewSetAll, // Note that this is not All, but is All SOS views.
    kViewSetDefault,
    kViewSetInitial,
    kViewSetAlwaysOn,
    kViewSetV0,
    kViewSetRequiredForBackup,
    kViewSetCKKS,
} ViewSetKind;

CFMutableSetRef SOSViewCopyViewSet(ViewSetKind setKind);



CFSetRef SOSViewsGetV0ViewSet(void);
CFSetRef SOSViewsGetV0SubviewSet(void);
CFSetRef SOSViewsGetV0BackupViewSet(void);
CFSetRef SOSViewsGetV0BackupBagViewSet(void);

CFSetRef SOSViewsGetUserVisibleSet(void);
bool SOSViewsIsV0Subview(CFStringRef viewName);

bool SOSViewInSOSSystem(CFStringRef view);
bool SOSViewHintInSOSSystem(CFStringRef viewHint);
bool SOSViewHintInCKKSSystem(CFStringRef viewHint);

// Basic interfaces to change and query views
SOSViewResultCode SOSViewsEnable(SOSPeerInfoRef pi, CFStringRef viewname, CFErrorRef *error);
bool SOSViewSetEnable(SOSPeerInfoRef pi, CFSetRef viewSet);
SOSViewResultCode SOSViewsDisable(SOSPeerInfoRef pi, CFStringRef viewname, CFErrorRef *error);
bool SOSViewSetDisable(SOSPeerInfoRef pi, CFSetRef viewSet);
SOSViewResultCode SOSViewsQuery(SOSPeerInfoRef pi, CFStringRef viewname, CFErrorRef *error);

CFSetRef SOSViewsGetAllCurrent(void);
void SOSViewsForEachDefaultEnabledViewName(void (^operation)(CFStringRef viewName));

CFSetRef SOSViewCreateSetFromBitmask(uint64_t bitmask);

// Test constraints
void SOSViewsSetTestViewsSet(CFSetRef testViewNames);


bool SOSViewSetIntersectsV0(CFSetRef theSet);
bool SOSPeerInfoV0ViewsEnabled(SOSPeerInfoRef pi);
bool SOSPeerInfoHasUserVisibleViewsEnabled(SOSPeerInfoRef pi);

bool SOSPeerInfoIsViewPermitted(SOSPeerInfoRef peerInfo, CFStringRef viewName);

const char *SOSViewsXlateAction(SOSViewActionCode action);
/* CFSet <-> XPC functions */
xpc_object_t CreateXPCObjectWithCFSetRef(CFSetRef setref, CFErrorRef *error);

__END_DECLS

#endif /* defined(_sec_SOSViews_) */
