/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include "trust_settings_impexp.h"
#include "security_tool.h"
#include <Security/Security.h>
#include <Security/SecTrustSettings.h>
#include <errno.h>
#include <unistd.h>
#include <security_cdsa_utils/cuFileIo.h>
#include <CoreFoundation/CoreFoundation.h>
#include <utilities/fileIo.h>

extern int trust_settings_export(int argc, char * const *argv)
{
	extern char *optarg;
	extern int optind;
	OSStatus ortn;
	int arg;
	CFDataRef settings = NULL;
	SecTrustSettingsDomain domain = kSecTrustSettingsDomainUser;
	int rtn;
	char *settingsFile = NULL;
	unsigned len;

	if(argc < 2) {
		return SHOW_USAGE_MESSAGE;
	}

	optind = 1;
	while ((arg = getopt(argc, argv, "dsh")) != -1) {
		switch (arg) {
			case 'd':
				domain = kSecTrustSettingsDomainAdmin;
				break;
			case 's':
				domain = kSecTrustSettingsDomainSystem;
				break;
			default:
				return SHOW_USAGE_MESSAGE;
		}
	}
	if(optind != (argc - 1)) {
		/* no args left for settings file */
		return SHOW_USAGE_MESSAGE;
	}
	settingsFile = argv[optind];

	ortn = SecTrustSettingsCreateExternalRepresentation(domain, &settings);
	if(ortn) {
		cssmPerror("SecTrustSettingsCreateExternalRepresentation", ortn);
		return 1;
	}
	len = (unsigned) CFDataGetLength(settings);
	rtn = writeFile(settingsFile, CFDataGetBytePtr(settings), len);
	if(rtn) {
		fprintf(stderr, "Error (%d) writing %s.\n", rtn, settingsFile);
	}
	else if(!do_quiet) {
		fprintf(stdout, "...Trust Settings exported successfully.\n");
	}
	CFRelease(settings);
	return rtn;
}

extern int trust_settings_import(int argc, char * const *argv)
{
	extern char *optarg;
	extern int optind;
	OSStatus ortn;
	int arg;
	char *settingsFile = NULL;
	unsigned char *settingsData = NULL;
	size_t settingsLen = 0;
	CFDataRef settings = NULL;
	SecTrustSettingsDomain domain = kSecTrustSettingsDomainUser;
	int rtn;

	if(argc < 2) {
		return SHOW_USAGE_MESSAGE;
	}

	optind = 1;
	while ((arg = getopt(argc, argv, "dh")) != -1) {
		switch (arg) {
			case 'd':
				domain = kSecTrustSettingsDomainAdmin;
				break;
			default:
				return SHOW_USAGE_MESSAGE;
		}
	}
	if(optind != (argc - 1)) {
		/* no args left for settings file */
		return SHOW_USAGE_MESSAGE;
	}
	settingsFile = argv[optind];
	rtn = readFileSizet(settingsFile, &settingsData, &settingsLen);
	if(rtn) {
		fprintf(stderr, "Error (%d) reading %s.\n", rtn, settingsFile);
		return 1;
	}
	settings = CFDataCreate(NULL, (const UInt8 *)settingsData, settingsLen);
	free(settingsData);
	ortn = SecTrustSettingsImportExternalRepresentation(domain, settings);
	CFRelease(settings);
	if(ortn) {
		cssmPerror("SecTrustSettingsImportExternalRepresentation", ortn);
		rtn = 1;
	}
	else if(!do_quiet) {
		fprintf(stdout, "...Trust Settings imported successfully.\n");
		rtn = 0;
	}
	return rtn;
}

