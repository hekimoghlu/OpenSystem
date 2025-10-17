/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
#import <Foundation/Foundation.h>

#if TARGET_OS_IOS && !TARGET_OS_VISION

#import <MobileInBoxUpdate/MobileInBoxUpdate.h>
#import <MobileInBoxUpdate/MIBUClient.h>
#import <Security/SecIdentityPriv.h>
#import "MIBConfigurationAccess.h"
#import "myCFUtil.h"
#import "EAPLog.h"

static MIBUClient *
GetMIBUClient(void)
{
    static dispatch_once_t	once;
    static MIBUClient 		*mibuClient = nil;

    dispatch_once(&once, ^{
	mibuClient = [[MIBUClient alloc] init];
    });
    return (mibuClient);
}

static MIBEAPConfigurationRef
MIBConfigurationConvertToMIBEAPConfiguration(MIBUEAPConfiguartion *configuration) {
    MIBEAPConfigurationRef eapConfig = NULL;
    if (configuration) {
	if (CFGetTypeID((CFTypeRef)[configuration.tlsCertificateChain firstObject]) != SecCertificateGetTypeID()) {
	    EAPLOG_FL(LOG_ERR, "received invalid client certificate from MIB");
	    return NULL;
	}
	if (CFGetTypeID(configuration.tlsKey) != SecKeyGetTypeID()) {
	    EAPLOG_FL(LOG_ERR, "received invalid client private key from MIB");
	    return NULL;
	}
	eapConfig = (MIBEAPConfigurationRef)malloc(sizeof(*eapConfig));
	bzero(eapConfig, sizeof(*eapConfig));
	SecCertificateRef leaf = (__bridge SecCertificateRef)[configuration.tlsCertificateChain firstObject];
	eapConfig->tlsClientIdentity = SecIdentityCreate(kCFAllocatorDefault, leaf, (SecKeyRef)configuration.tlsKey);
	if (configuration.tlsCertificateChain.count > 1) {
	    CFMutableArrayRef certs = CFArrayCreateMutableCopy(NULL, configuration.tlsCertificateChain.count, (__bridge CFArrayRef)configuration.tlsCertificateChain);
	    CFArrayRemoveValueAtIndex(certs, 0);
	    eapConfig->tlsClientCertificateChain = certs;
	}
    }
    return eapConfig;
}

void
MIBConfigurationAccessFetchEAPConfiguration(MIBConfigurationAccessCallback callback, void *context) {
    @autoreleasepool {
	MIBUClient *mibuClient = GetMIBUClient();
	[mibuClient eapConfigurationWithCompletion:^(MIBUEAPConfiguartion *configuration, NSError *error) {
	    MIBEAPConfigurationRef eapConfig = NULL;
	    if (error) {
		EAPLOG_FL(LOG_ERR, "failed to fetch EAP configuration from MIB, error: %@", error);
	    } else {
		eapConfig = MIBConfigurationConvertToMIBEAPConfiguration(configuration);
	    }
	    dispatch_async(dispatch_get_main_queue(), ^{
		if (callback) {
		    callback(context, eapConfig);
		}
	    });
	}];
    }
}

bool
MIBConfigurationAccessIsInBoxUpdateMode(void) {
    @autoreleasepool {
	MIBUClient *mibuClient = GetMIBUClient();
	return ([mibuClient isInBoxUpdateMode:nil] == YES);
    }
}

#endif /* TARGET_OS_IOS && !TARGET_OS_VISION */
