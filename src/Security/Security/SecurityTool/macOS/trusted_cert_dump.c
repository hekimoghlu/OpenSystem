/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#include "security_tool.h"

#include "trusted_cert_dump.h"
#include "trusted_cert_utils.h"

#include <errno.h>
#include <unistd.h>
#include <Security/Security.h>
#include <Security/cssmapple.h>
#include <Security/SecTrustSettings.h>
#include <Security/oidsalg.h>
#include <security_cdsa_utils/cuFileIo.h>
#include <CoreFoundation/CoreFoundation.h>

/*
 * Display a Trust Settings array as obtained from
 * SecTrustSettingsCopyTrustSettings().
 */
static int displayTrustSettings(
	CFArrayRef	trustSettings)
{
	/* must always be there though it may be empty */
	if(trustSettings == NULL) {
		fprintf(stderr, "***displayTrustSettings: missing trust settings array");
		return -1;
	}
	if(CFGetTypeID(trustSettings) != CFArrayGetTypeID()) {
		fprintf(stderr, "***displayTrustSettings: malformed trust settings array");
		return -1;
	}

	int ourRtn = 0;
	CFIndex numUseConstraints = CFArrayGetCount(trustSettings);
	indentIncr();
	indent(); printf("Number of trust settings : %ld\n", (long)numUseConstraints);
	OSStatus ortn;
	SecPolicyRef certPolicy;
	SecTrustedApplicationRef certApp;
	CFDictionaryRef ucDict;
	CFStringRef policyStr;
	CFNumberRef cfNum;
	CFIndex ucDex;

	/* grind thru the trust settings dictionaries */
	for(ucDex=0; ucDex<numUseConstraints; ucDex++) {
		indent(); printf("Trust Setting %ld:\n", (long)ucDex);
		indentIncr();

		ucDict = (CFDictionaryRef)CFArrayGetValueAtIndex(trustSettings, ucDex);
		if(CFGetTypeID(ucDict) != CFDictionaryGetTypeID()) {
			fprintf(stderr, "***displayTrustSettings: malformed usage constraints dictionary");
			ourRtn = -1;
			goto nextAp;
		}

		/* policy - optional */
		certPolicy = (SecPolicyRef)CFDictionaryGetValue(ucDict, kSecTrustSettingsPolicy);
		if(certPolicy != NULL) {
			if(CFGetTypeID(certPolicy) != SecPolicyGetTypeID()) {
				fprintf(stderr, "***displayTrustSettings: malformed certPolicy");
				ourRtn = -1;
				goto nextAp;
			}
			CSSM_OID policyOid;
			ortn = SecPolicyGetOID(certPolicy, &policyOid);
			if(ortn) {
				cssmPerror("SecPolicyGetOID", ortn);
				ourRtn = -1;
				goto nextAp;
			}
			indent(); printf("Policy OID            : %s\n",
					oidToOidString(&policyOid));
		}

		/* app - optional  */
		certApp = (SecTrustedApplicationRef)CFDictionaryGetValue(ucDict,
			kSecTrustSettingsApplication);
		if(certApp != NULL) {
			if(CFGetTypeID(certApp) != SecTrustedApplicationGetTypeID()) {
				fprintf(stderr, "***displayTrustSettings: malformed certApp");
				ourRtn = -1;
				goto nextAp;
			}
			CFDataRef appPath = NULL;
			ortn = SecTrustedApplicationCopyData(certApp, &appPath);
			if(ortn) {
				cssmPerror("SecTrustedApplicationCopyData", ortn);
				ourRtn = -1;
				goto nextAp;
			}
			indent(); printf("Application           : %s", CFDataGetBytePtr(appPath));
			printf("\n");
			CFRelease(appPath);
		}

		/* policy string */
		policyStr = (CFStringRef)CFDictionaryGetValue(ucDict, kSecTrustSettingsPolicyString);
		if(policyStr != NULL) {
			if(CFGetTypeID(policyStr) != CFStringGetTypeID()) {
				fprintf(stderr, "***displayTrustSettings: malformed policyStr");
				ourRtn = -1;
				goto nextAp;
			}
			indent(); printf("Policy String         : ");
			printCfStr(policyStr); printf("\n");
		}

		/* Allowed error */
		cfNum = (CFNumberRef)CFDictionaryGetValue(ucDict, kSecTrustSettingsAllowedError);
		if(cfNum != NULL) {
			if(CFGetTypeID(cfNum) != CFNumberGetTypeID()) {
				fprintf(stderr, "***displayTrustSettings: malformed allowedError");
				ourRtn = -1;
				goto nextAp;
			}
			indent(); printf("Allowed Error         : ");
			printCssmErr(cfNum); printf("\n");
		}

		/* ResultType */
		cfNum = (CFNumberRef)CFDictionaryGetValue(ucDict, kSecTrustSettingsResult);
		if(cfNum != NULL) {
			if(CFGetTypeID(cfNum) != CFNumberGetTypeID()) {
				fprintf(stderr, "***displayTrustSettings: malformed ResultType");
				ourRtn = -1;
				goto nextAp;
			}
			indent(); printf("Result Type           : ");
			printResultType(cfNum); printf("\n");
		}

		/* key usage */
		cfNum = (CFNumberRef)CFDictionaryGetValue(ucDict, kSecTrustSettingsKeyUsage);
		if(cfNum != NULL) {
			if(CFGetTypeID(cfNum) != CFNumberGetTypeID()) {
				fprintf(stderr, "***displayTrustSettings: malformed keyUsage");
				ourRtn = -1;
				goto nextAp;
			}
			indent(); printf("Key Usage             : ");
			printKeyUsage(cfNum); printf("\n");
		}

	nextAp:
		indentDecr();
	}
	indentDecr();
	return ourRtn;
}

int
trusted_cert_dump(int argc, char * const *argv)
{
	CFArrayRef certArray = NULL;
	OSStatus ortn = noErr;
	CFIndex numCerts;
	CFIndex dex;
	CFArrayRef trustSettings;
	int ourRtn = 0;
	SecTrustSettingsDomain domain = kSecTrustSettingsDomainUser;

	extern char *optarg;
	extern int optind;
	int arg;

	optind = 1;
	while ((arg = getopt(argc, argv, "sdh")) != -1) {
		switch (arg) {
			case 's':
				domain = kSecTrustSettingsDomainSystem;
				break;
			case 'd':
				domain = kSecTrustSettingsDomainAdmin;
				break;
			default:
			case 'h':
				return SHOW_USAGE_MESSAGE;
		}
	}

	if(optind != argc) {
		return SHOW_USAGE_MESSAGE;
	}

	ortn = SecTrustSettingsCopyCertificates(domain, &certArray);
	if(ortn) {
		cssmPerror("SecTrustSettingsCopyCertificates", ortn);
		return 1;
	}
	numCerts = CFArrayGetCount(certArray);
	printf("Number of trusted certs = %ld\n", (long)numCerts);

	for(dex=0; dex<numCerts; dex++) {
		SecCertificateRef certRef =
				(SecCertificateRef)CFArrayGetValueAtIndex(certArray, dex);
		if(CFGetTypeID(certRef) != SecCertificateGetTypeID()) {
			fprintf(stderr, "***Bad CFGetTypeID for cert %ld\n", (long)dex);
			ourRtn = -1;
			break;
		}

		/* always print the cert's label */
		printf("Cert %ld: ", dex);
		printCertLabel(certRef);
		printf("\n");

		/* see if the cert has any usage constraints (it should!) */
		ortn = SecTrustSettingsCopyTrustSettings(certRef, domain, &trustSettings);
		if(ortn) {
			cssmPerror("SecTrustSettingsCopyTrustSettings", ortn);
			ourRtn = -1;
			continue;
		}
		if(displayTrustSettings(trustSettings)) {
			ourRtn = -1;
		}
	}
	CFRelease(certArray);

	return ourRtn;
}
