/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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


#include "keychain/SecureObjectSync/SOSTransport.h"
#include "keychain/SecureObjectSync/SOSTransportKeyParameter.h"
#include "keychain/SecureObjectSync/SOSKVSKeys.h"
#include "keychain/securityd/SOSCloudCircleServer.h"
#include <utilities/SecCFWrappers.h>
#include "keychain/SecureObjectSync/SOSAccountPriv.h"
#include "keychain/SecureObjectSync/CKBridge/SOSCloudKeychainClient.h"

@implementation CKKeyParameter

@synthesize account = account;

-(bool) SOSTransportKeyParameterHandleKeyParameterChanges:(CKKeyParameter*) transport  data:(CFDataRef) data err:(CFErrorRef) error
{
    return SOSAccountHandleParametersChange(account, data, &error);
}

-(SOSAccount*) SOSTransportKeyParameterGetAccount:(CKKeyParameter*) transport
{
    return account;
}

-(CFIndex) SOSTransportKeyParameterGetTransportType:(CKKeyParameter*) transport err:(CFErrorRef *)error
{
    return kKVS;
}

-(id) initWithAccount:(SOSAccount*) acct
{
    if ((self = [super init])) {
        self.account = acct;
        SOSRegisterTransportKeyParameter(self);
    }
    return self;
}

-(bool) SOSTransportKeyParameterKVSAppendKeyInterests:(CKKeyParameter*)transport ak:(CFMutableArrayRef)alwaysKeys firstUnLock:(CFMutableArrayRef)afterFirstUnlockKeys unlocked:(CFMutableArrayRef) unlockedKeys err:(CFErrorRef *)error
{
    CFArrayAppendValue(alwaysKeys, kSOSKVSKeyParametersKey);
    CFArrayAppendValue(alwaysKeys, kSOSKVSOfficialDSIDKey);
    return true;
}

static bool SOSTransportKeyParameterKVSUpdateKVS(CFDictionaryRef changes, CFErrorRef *error){
    CloudKeychainReplyBlock log_error = ^(CFDictionaryRef returnedValues __unused, CFErrorRef block_error) {
        if (block_error) {
            secerror("Error putting: %@", block_error);
        }
    };

    SOSCloudKeychainPutObjectsInCloud(changes, dispatch_get_global_queue(SOS_TRANSPORT_PRIORITY, 0), log_error);
    return true;
}

-(bool) SOSTransportKeyParameterPublishCloudParameters:(CKKeyParameter*) transport data:(CFDataRef)newParameters err:(CFErrorRef*) error
{
    if(newParameters) {
        secnotice("circleOps", "Publishing Cloud Parameters");
    } else {
        secnotice("circleOps", "Tried to publish nil Cloud Parameters");
        (void) SecRequirementError(newParameters != NULL, error, CFSTR("Tried to publish nil Cloud Parameters"));
        return false;
    }

    bool waitForeverForSynchronization = false;
    CFDictionaryRef changes = NULL;
    CFDataRef timeData = NULL;
    CFMutableStringRef timeDescription = CFStringCreateMutableCopy(kCFAllocatorDefault, 0, CFSTR("["));
    CFAbsoluteTime currentTimeAndDate = CFAbsoluteTimeGetCurrent();

    withStringOfAbsoluteTime(currentTimeAndDate, ^(CFStringRef decription) {
        CFStringAppend(timeDescription, decription);
    });
    CFStringAppend(timeDescription, CFSTR("]"));

    timeData = CFStringCreateExternalRepresentation(NULL,timeDescription,
                                                    kCFStringEncodingUTF8, '?');

    CFMutableDataRef timeAndKeyParametersMutable = CFDataCreateMutable(kCFAllocatorDefault, CFDataGetLength(timeData) + CFDataGetLength(newParameters));
    CFDataAppend(timeAndKeyParametersMutable, timeData);
    CFDataAppend(timeAndKeyParametersMutable, newParameters);
    CFDataRef timeAndKeyParameters = CFDataCreateCopy(kCFAllocatorDefault, timeAndKeyParametersMutable);

    CFStringRef ourPeerID = (__bridge CFStringRef)account.peerID;

    if(ourPeerID != NULL){
        CFStringRef keyParamKey = SOSLastKeyParametersPushedKeyCreateWithPeerID(ourPeerID);

        changes = CFDictionaryCreateForCFTypes(kCFAllocatorDefault,
                                               kSOSKVSKeyParametersKey, newParameters,
                                               keyParamKey, timeAndKeyParameters,
                                               NULL);
        CFReleaseNull(keyParamKey);
    }
    else
    {
        CFStringRef keyParamKeyWithAccount = SOSLastKeyParametersPushedKeyCreateWithAccountGestalt(account);
        changes = CFDictionaryCreateForCFTypes(kCFAllocatorDefault,
                                               kSOSKVSKeyParametersKey, newParameters,
                                               keyParamKeyWithAccount, timeAndKeyParameters,
                                               NULL);
        CFReleaseNull(keyParamKeyWithAccount);
    }
    bool success = SOSTransportKeyParameterKVSUpdateKVS(changes, error);

    sync_the_last_data_to_kvs((__bridge CFTypeRef)(account), waitForeverForSynchronization);
    CFReleaseNull(changes);
    CFReleaseNull(timeData);
    CFReleaseNull(timeAndKeyParameters);
    CFReleaseNull(timeAndKeyParametersMutable);
    CFReleaseNull(timeDescription);

    return success;
}

@end

