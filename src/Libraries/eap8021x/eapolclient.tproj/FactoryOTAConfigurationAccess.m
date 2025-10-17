/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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

#import <FactoryOTAEAPClient/FactoryOTAEAPClient.h>
#import <Security/SecIdentityPriv.h>
#import "FactoryOTAConfigurationAccess.h"
#import "myCFUtil.h"
#import "EAPLog.h"

static FactoryOTAEAPClient *
GetFactoryOTAEAPClient(void)
{
    static dispatch_once_t		once;
    static FactoryOTAEAPClient 		*factoryOTAClient = nil;

    dispatch_once(&once, ^{
	factoryOTAClient = [[FactoryOTAEAPClient alloc] init];
    });
    return (factoryOTAClient);
}

static Boolean
CopyFOTAEAPConfiguration(FactoryOTAEAPConfiguration *configuration, FOTAEAPConfigurationRef eapConfig) {
    if (configuration) {
	if (CFGetTypeID((CFTypeRef)[configuration.tlsCertificateChain firstObject]) != SecCertificateGetTypeID()) {
	    EAPLOG_FL(LOG_ERR, "received invalid client certificate from Factory OTA Client");
	    return FALSE;
	}
	if (CFGetTypeID(configuration.tlsKey) != SecKeyGetTypeID()) {
	    EAPLOG_FL(LOG_ERR, "received invalid client private key from Factory OTA Client");
	    return FALSE;
	}
	bzero(eapConfig, sizeof(*eapConfig));
	SecCertificateRef leaf = (__bridge SecCertificateRef)[configuration.tlsCertificateChain firstObject];
	eapConfig->tlsClientIdentity = SecIdentityCreate(kCFAllocatorDefault, leaf, (SecKeyRef)configuration.tlsKey);
	if (configuration.tlsCertificateChain.count > 1) {
	    CFMutableArrayRef certs = CFArrayCreateMutableCopy(NULL, configuration.tlsCertificateChain.count, (__bridge CFArrayRef)configuration.tlsCertificateChain);
	    CFArrayRemoveValueAtIndex(certs, 0);
	    eapConfig->tlsClientCertificateChain = certs;
	}
    }
    return TRUE;
}

void
FactoryOTAConfigurationAccessFetchEAPConfiguration(FactoryOTAConfigurationAccessCallback callback, void *context) {
    @autoreleasepool {
	FactoryOTAEAPClient *factoryOTAClient = GetFactoryOTAEAPClient();
	[factoryOTAClient eapConfigurationWithCompletion:^(FactoryOTAEAPConfiguration *configuration, NSError *error) {
	    FOTAEAPConfiguration eapConfig;
	    Boolean success = FALSE;
	    if (error) {
		EAPLOG_FL(LOG_ERR, "failed to fetch EAP configuration from Factory OTA Client, error: %@", error);
	    } else {
		EAPLOG_FL(LOG_NOTICE, "received EAP configuration from Factory OTA Client");
		success = CopyFOTAEAPConfiguration(configuration, &eapConfig);
	    }
	    if (callback) {
		FOTAEAPConfigurationRef eapConfigRef = success ? &eapConfig : NULL;
		callback(context, eapConfigRef);
		if (eapConfigRef != NULL) {
		    my_CFRelease(&eapConfigRef->tlsClientIdentity);
		    my_CFRelease(&eapConfigRef->tlsClientCertificateChain);
		}
	    }
	}];
    }
}

bool
FactoryOTAConfigurationAccessIsInFactoryMode(void) {
    @autoreleasepool {
	FactoryOTAEAPClient *factoryOTAClient = GetFactoryOTAEAPClient();
	return ([factoryOTAClient isInFactoryMode] == YES);
    }
}

#endif /* TARGET_OS_IOS && !TARGET_OS_VISION */
